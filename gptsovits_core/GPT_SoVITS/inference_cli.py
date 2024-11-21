
# python inference_cli.py --gpt_model "E:\GPT-SoVITS-main\GPT-SoVITS-v2-240821\GPT_weights_v2\YQ_V2_V0.2-e15.ckpt" --sovits_model "E:\GPT-SoVITS-main\GPT-SoVITS-v2-240821\SoVITS_weights_v2\YQ_V2_V0.2_e8_s272.pth" --ref_audio "E:\ML_Data_Collection\GPT-SoVITS\YaQian05\TestAU.wav" --ref_text "E:\ML_Data_Collection\GPT-SoVITS\YaQian05\Wav_text.txt" --ref_language 中文 --target_text "E:\ML_Data_Collection\GPT-SoVITS\YaQian05\Test_target.txt" --target_language 中文 --output_path "E:\GPT-SoVITS-main\GPT-SoVITS-v2-240821\output"
#python gptsovits_core/GPT_SoVITS/inference_cli.py --gpt_model "E:\Private_Assistant\Models_Pretrained\GPT_SoVITS\en\GPT_weights_v2\Yae_Miko_V2_Genshin5.1.ckpt" --sovits_model "E:\Private_Assistant\Models_Pretrained\GPT_SoVITS\en\SoVITS_weights_v2\Yae_Miko_V2_Genshin5.1.pth" --ref_audio "E:\Private_Assistant\References\en\Yae_Miko_V2_Genshin5.1.wav" --ref_text "E:\Private_Assistant\References\en\Yae_Miko_V2_Genshin5.1.txt" --ref_language 英文 --target_text "E:\Private_Assistant\References\en\t.txt" --target_language 英文 --output_path "E:\Private_Assistant\temp"


import argparse
import os
import soundfile as sf

from tools.i18n.i18n import I18nAuto
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()

def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, output_path, how_to_cut=i18n("不切")):
    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # Read target text
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # Synthesize audio
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path, 
                                   prompt_text=ref_text, 
                                   prompt_language=i18n(ref_language), 
                                   text=target_text, 
                                   text_language=i18n(target_language), top_p=1, temperature=1, how_to_cut=i18n("按英文句号.切"))
    
    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument('--gpt_model', required=True, help="Path to the GPT model file")
    parser.add_argument('--sovits_model', required=True, help="Path to the SoVITS model file")
    parser.add_argument('--ref_audio', required=True, help="Path to the reference audio file")
    parser.add_argument('--ref_text', required=True, help="Path to the reference text file")
    parser.add_argument('--ref_language', required=True, choices=["中文", "英文", "日文"], help="Language of the reference audio")
    parser.add_argument('--target_text', required=True, help="Path to the target text file")
    parser.add_argument('--target_language', required=True, choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], help="Language of the target text")
    parser.add_argument('--output_path', required=True, help="Path to the output directory")

    args = parser.parse_args()

    synthesize(args.gpt_model, args.sovits_model, args.ref_audio, args.ref_text, args.ref_language, args.target_text, args.target_language, args.output_path)

if __name__ == '__main__':
    main()

