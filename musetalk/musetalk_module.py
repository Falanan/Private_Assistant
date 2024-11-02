import os
import cv2
import torch
import numpy as np
import glob
import pickle
import shutil
from tqdm import tqdm
import copy
from omegaconf import OmegaConf
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model

class InferenceConfig:
    def __init__(self, video_path, audio_path, bbox_shift=0, result_dir='Inference_results/MuseTalk', 
                 fps=25, batch_size=8, output_vid_name='YaQian_refer.mp4', 
                 use_saved_coord=False, use_float16=False):
        self.video_path = video_path
        self.audio_path = audio_path
        self.bbox_shift = bbox_shift
        self.result_dir = result_dir
        self.fps = fps
        self.batch_size = batch_size
        self.output_vid_name = output_vid_name
        self.use_saved_coord = use_saved_coord
        self.use_float16 = use_float16

class MuseTalkInference:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=self.device)

        # Load models
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()

        # Half precision
        if self.config.use_float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

    def process_video(self):

        video_path = self.config.video_path
        audio_path = self.config.audio_path
        bbox_shift = self.config.bbox_shift
        
        # Prepare paths and directories
        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        result_img_save_path = os.path.join(self.config.result_dir, output_basename)
        crop_coord_save_path = os.path.join(result_img_save_path, f"{input_basename}.pkl")
        os.makedirs(result_img_save_path, exist_ok=True)

        if self.config.output_vid_name is None:
            output_vid_name = os.path.join(self.config.result_dir, f"{output_basename}.mp4")
        else:
            output_vid_name = os.path.join(self.config.result_dir, self.config.output_vid_name)

        # Extract frames from video
        input_img_list, fps = self.extract_frames(video_path)

        # Extract audio feature
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)

        # Preprocess images
        coord_list, frame_list = self.preprocess_images(input_img_list, crop_coord_save_path, bbox_shift)

        # Prepare latent space representation
        input_latent_list = self.get_latents(coord_list, frame_list)

        # Generate the talking video frames
        res_frame_list = self.generate_video_frames(whisper_chunks, input_latent_list)

        # Combine generated frames into a full video
        self.create_video(res_frame_list, coord_list, frame_list, fps, result_img_save_path, output_vid_name, audio_path)

    def extract_frames(self, video_path):
        if get_file_type(video_path) == "video":
            save_dir_full = os.path.join(self.config.result_dir, os.path.basename(video_path).split('.')[0])
            os.makedirs(save_dir_full, exist_ok=True)
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        elif get_file_type(video_path) == "image":
            input_img_list = [video_path]
            fps = self.config.fps
        elif os.path.isdir(video_path):
            input_img_list = sorted(glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]')),
                                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = self.config.fps
        else:
            raise ValueError(f"{video_path} should be a video file, an image file, or a directory of images")
        return input_img_list, fps

    def preprocess_images(self, input_img_list, crop_coord_save_path, bbox_shift):
        if os.path.exists(crop_coord_save_path) and self.config.use_saved_coord:
            with open(crop_coord_save_path, 'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
        return coord_list, frame_list

    def get_latents(self, coord_list, frame_list):
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        return input_latent_list

    def generate_video_frames(self, whisper_chunks, input_latent_list):
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        video_num = len(whisper_chunks)
        batch_size = self.config.batch_size
        gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
        res_frame_list = []
        for whisper_batch, latent_batch in tqdm(gen, total=int(np.ceil(float(video_num) / batch_size))):
            audio_feature_batch = torch.from_numpy(whisper_batch).to(device=self.unet.device, dtype=self.unet.model.dtype)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
            audio_feature_batch = self.pe(audio_feature_batch)
            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
            res_frame_list.extend(self.vae.decode_latents(pred_latents))
        return res_frame_list

    def create_video(self, res_frame_list, coord_list, frame_list, fps, result_img_save_path, output_vid_name, audio_path):
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list[i % len(coord_list)]
            ori_frame = copy.deepcopy(frame_list[i % len(frame_list)])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                combine_frame = get_image(ori_frame, res_frame, bbox)
                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
            except Exception as e:
                continue

        # Combine frames to video
        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        os.system(cmd_img2video)
        cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
        os.system(cmd_combine_audio)
        os.remove("temp.mp4")
        shutil.rmtree(result_img_save_path)

def run_inference_from_args():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    # parser.add_argument("--bbox_shift", type=int, default=0)
    # parser.add_argument("--result_dir", default='./results', help="path to output")
    # parser.add_argument("--fps", type=int, default=25)
    # parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--output_vid_name", type=str, default=None)
    # parser.add_argument("--use_saved_coord", action="store_true", help="use saved coordinates")
    # parser.add_argument("--use_float16", action="store_true", help="use float16 for faster inference")

    # args = parser.parse_args()
        # print(args, "-------------------------------------------------------------------------------------------------------------------------")
    config = InferenceConfig(
            video_path="Inference_results/Originals/Yaqian_original_image_preprocessed.jpg",
            audio_path="Inference_results/Originals/YaQian_Refer.wav",
            bbox_shift=0,
            result_dir='Inference_results/MuseTalk',
            fps=25,
            batch_size=8,
            output_vid_name='YaQian_refer.mp4',
            use_saved_coord=False,
            use_float16=True
        )
    inference = MuseTalkInference(config)
    inference.process_video()

if __name__ == "__main__":
    run_inference_from_args()
