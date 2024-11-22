import os
import cv2
import numpy as np
import subprocess
import torch
from tqdm import tqdm
import audio
from models import Wav2Lip
import face_detection
from glob import glob

class Wav2LipInference:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(config.checkpoint_path)

    def load_model(self, path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        return model.to(self.device).eval()

    def preprocess_audio(self):
        if not self.config.audio.endswith('.wav'):
            command = f'ffmpeg -y -loglevel quiet -i {self.config.audio} -strict -2 wav2lip_core/temp/temp.wav'
            subprocess.call(command, shell=True)
            self.config.audio = 'wav2lip_core/temp/temp.wav'

        wav = audio.load_wav(self.config.audio, 16000)
        mel = audio.melspectrogram(wav)
        return mel

    def preprocess_video(self):
        if os.path.isfile(self.config.face) and self.config.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            frames = [cv2.imread(self.config.face)]
            fps = self.config.fps
        else:
            video_stream = cv2.VideoCapture(self.config.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            print('Reading video frames...')
            frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.config.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1] // self.config.resize_factor, frame.shape[0] // self.config.resize_factor))
                if self.config.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frames.append(frame)
        return frames, fps

    def run_inference(self):
        mel = self.preprocess_audio()
        frames, fps = self.preprocess_video()
        mel_chunks = self.create_mel_chunks(mel, fps)

        full_frames = frames[:len(mel_chunks)]
        batch_size = self.config.wav2lip_batch_size
        gen = self.datagen(full_frames, mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(len(mel_chunks) / batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('wav2lip_core/temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()
        self.combine_audio()

    def create_mel_chunks(self, mel, fps):
        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > mel.shape[1]:
                mel_chunks.append(mel[:, -16:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + 16])
            i += 1
        return mel_chunks

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        face_det_results = self.face_detect(frames)
        for i, m in enumerate(mels):
            idx = 0 if self.config.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()
            face = cv2.resize(face, (96, 96))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.config.wav2lip_batch_size:
                yield self.prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if img_batch:
            yield self.prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)

    def prepare_batch(self, img_batch, mel_batch, frame_batch, coords_batch):
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, img_batch.shape[1] // 2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, (len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1))
        return img_batch, mel_batch, frame_batch, coords_batch

    # def face_detect(self, images):
    #     detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=self.device)
    #     predictions = [detector.get_detections_for_batch(np.array(images[i:i + self.config.face_det_batch_size]))
    #                    for i in range(0, len(images), self.config.face_det_batch_size)]
    #     results = []
    #     pady1, pady2, padx1, padx2 = self.config.pads
    #     for rect, image in zip(predictions, images):
    #         y1, y2, x1, x2 = max(0, rect[1] - pady1), min(image.shape[0], rect[3] + pady2), max(0, rect[0] - padx1), min(image.shape[1], rect[2] + padx2)
    #         results.append([image[y1:y2, x1:x2], (y1, y2, x1, x2)])
    #     return results
    
    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
    

    def face_detect(self, images):
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                flip_input=False, device=self.device)

        batch_size = self.config.face_det_batch_size
        
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.config.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('wav2lip_core/temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.config.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results 
    def combine_audio(self):
        command = f'ffmpeg -y -loglevel quiet -i {self.config.audio} -i wav2lip_core/temp/result.avi {self.config.outfile}'
        subprocess.call(command, shell=True)

class InferenceConfig:
    def __init__(self, checkpoint_path, face, audio, outfile='output/result_voice.mp4', static=False, fps=25.0,
                 pads=[0, 10, 0, 0], face_det_batch_size=16, wav2lip_batch_size=128, resize_factor=1,
                 crop=[0, -1, 0, -1], box=[-1, -1, -1, -1], rotate=False, nosmooth=False):
        self.checkpoint_path = checkpoint_path
        self.face = face
        self.audio = audio
        self.outfile = outfile
        self.static = static
        self.fps = fps
        self.pads = pads
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.resize_factor = resize_factor
        self.crop = crop
        self.box = box
        self.rotate = rotate
        self.nosmooth = nosmooth

def run_inference_from_args():
    import argparse
    parser = argparse.ArgumentParser(description='Inference for lip-syncing videos with Wav2Lip.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--face', type=str, required=True, help='Path to video or image with face')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--outfile', type=str, default='output/result_voice.mp4', help='Output video file')
    parser.add_argument('--static', type=bool, default=False, help='Use only first video frame for inference')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS for static image input')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding for face detection')
    parser.add_argument('--face_det_batch_size', type=int, default=16, help='Batch size for face detection')
    parser.add_argument('--wav2lip_batch_size', type=int, default=128, help='Batch size for Wav2Lip model')
    parser.add_argument('--resize_factor', type=int, default=1, help='Reduce resolution by this factor')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop video to region')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Bounding box for face')
    parser.add_argument('--rotate', action='store_true', help='Rotate video by 90 degrees')
    parser.add_argument('--nosmooth', action='store_true', help='Disable smoothing for face detection')

    args = parser.parse_args()
    config = InferenceConfig(
        checkpoint_path=args.checkpoint_path,
        face=args.face,
        audio=args.audio,
        outfile=args.outfile,
        static=args.static,
        fps=args.fps,
        pads=args.pads,
        face_det_batch_size=args.face_det_batch_size,
        wav2lip_batch_size=args.wav2lip_batch_size,
        resize_factor=args.resize_factor,
        crop=args.crop,
        box=args.box,
        rotate=args.rotate,
        nosmooth=args.nosmooth,
    )
    inference = Wav2LipInference(config)
    inference.run_inference()

if __name__ == '__main__':
    # run_inference_from_args()
    # config = InferenceConfig(
    # checkpoint_path="Models_Pretrained/wav2lip/wav2lip_gan.pth",
    # face="Inference_results/Originals/Yae_Miko_Avatar.png",
    # audio="Inference_results/Originals/Yae_Miko_Refer.wav",
    # outfile="wav2lip_core/output/synced_video.mp4"
    # )

    config = InferenceConfig(
    checkpoint_path="Models_Pretrained/wav2lip/wav2lip_gan.pth",
    face="wav2lip_core/inputs/1012.mp4",
    audio="Inference_results/Originals/Yae_Miko_Refer.wav",
    outfile="wav2lip_core/output/synced_video.mp4",
    resize_factor=2
    )
    # Run inference
    inference = Wav2LipInference(config)
    inference.run_inference()   
