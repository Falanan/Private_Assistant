import sys
import os
import contextlib
import re
import nltk
import soundfile as sf
from chatglm.glm4_module import GLMChatbot
from pyqt_gui import MainWindow
from PyQt5.QtWidgets import QApplication


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "gptsovits_core")))
# Add additional subdirectories if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "gptsovits_core", "GPT_SoVITS")))
from gptsovits_core.GPT_SoVITS.tools.i18n.i18n import I18nAuto
from gptsovits_core.GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "wav2lip_core")))
from wav2lip_core.wav2lip_module import Wav2LipInference, InferenceConfig


_stdout_backup = sys.stdout
_stderr_backup = sys.stderr
i18n = I18nAuto()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

# GPT_SoVITS inference
def synthesize(ref_audio_path, ref_text_path, ref_language, target_text, target_language, output_path, how_to_cut=i18n("不切")):
    with open(ref_text_path, 'r', encoding='utf-8') as file:
            ref_text = file.read()
    # Synthesize audio
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path, 
                                   prompt_text=ref_text, 
                                   prompt_language=i18n(ref_language), 
                                   text=target_text, 
                                   text_language=i18n(target_language), top_p=1, temperature=1, how_to_cut = how_to_cut)
    
    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "t2s_output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")




def toggle_output(enable=True):
    """
    Toggle the output on or off.
    
    Parameters:
    - enable (bool): If True, output is enabled. If False, output is suppressed.
    """
    global _stdout_backup, _stderr_backup
    if enable:
        # Restore output
        sys.stdout = _stdout_backup
        sys.stderr = _stderr_backup
    else:
        # Suppress output
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')


def remove_emojis(s):
    # Emoji Unicode range
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', s)


def clean_text(text):
    no_emojy_text = remove_emojis(text)
    # Replace newline and escaped single quotes
    cleaned_text = no_emojy_text.replace("\\n", "").replace("\\'", "'").replace("\\t", "")
    # Optionally, replace multiple spaces with a single space for better formatting
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


def main():



    # This is the folder path that temporary store wav file and video file.
    temp_file_path = "temp"
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)
        print(f"Folder created: {temp_file_path}")

    # These are models paths
    models_parent_path = "Models_Pretrained"
    museTalk_model_path = os.path.join(models_parent_path, "MuseTalk")
    wav2lip_path = os.path.join(models_parent_path, "Wav2Lip", "wav2lip_gan.pth")
    gs_sovits_weight_path =  os.path.join(models_parent_path, "GPT_SoVITS")
    gs_gpt_weight_path = os.path.join(models_parent_path, "GPT_SoVITS")
    reference_avatar_path = os.path.join("References")
    reference_audio_path = os.path.join("References")
    reference_text_path = os.path.join("References")

    print("Please select your language:")
    print("1. English")
    print("2. Chinese")
    language = "en"

    # Get Language preference
    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice == "1":
            language = "en"
            gs_sovits_weight_path = os.path.join(gs_sovits_weight_path, "en", "SoVITS_weights_v2")
            gs_gpt_weight_path = os.path.join(gs_gpt_weight_path, "en", "GPT_weights_v2")
            reference_avatar_path = os.path.join(reference_avatar_path, "en")
            reference_audio_path = os.path.join(reference_audio_path, "en")
            reference_text_path = os.path.join(reference_text_path, "en")
            print("You selected English.")
            break
        elif choice == "2":
            language = "zh"
            gs_sovits_weight_path = os.path.join(gs_sovits_weight_path, "zh", "SoVITS_weights_v2")
            gs_gpt_weight_path = os.path.join(gs_gpt_weight_path, "zh", "GPT_weights_v2")
            reference_avatar_path = os.path.join(reference_avatar_path, "zh")
            reference_audio_path = os.path.join(reference_audio_path, "zh")
            reference_text_path = os.path.join(reference_text_path, "zh")
            print("You selected Chinese.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # Get model selection and update models path
    while True:
        with os.scandir(gs_sovits_weight_path) as entries:
            models_list = [entry.name for entry in entries if entry.is_file()]
        for k in range(1, len(models_list)+1):
            print(str(k)+":", models_list[k-1])
        choice = int(input("Enter your choice: "))
        if int(choice) <= len(models_list):
            model_name = models_list[choice-1].split('.pth')
            gs_sovits_weight_path = os.path.join(gs_sovits_weight_path, model_name[0] + ".pth")
            gs_gpt_weight_path = os.path.join(gs_gpt_weight_path, model_name[0] + ".ckpt")
            reference_avatar_path = os.path.join(reference_avatar_path, model_name[0] + ".png")
            reference_audio_path = os.path.join(reference_audio_path, model_name[0] + ".wav")
            reference_text_path = os.path.join(reference_text_path, model_name[0] + ".txt")

            print("You selected", model_name[choice-1],"model")
            break
        else:
            print("Invalid choice. Please enter again")



    if language == "en":
        # Pending all outputs when load models
        toggle_output(False)
        # nltk.download('averaged_perceptron_tagger_eng')

        #  Initilize GLM-4 model
        chatbot = GLMChatbot()

        # Initilize GPT-SoVITS model
        change_gpt_weights(gpt_path=gs_gpt_weight_path)
        change_sovits_weights(sovits_path=gs_sovits_weight_path)

        # Initilize Wav2Lip model
        wav2lip_config = InferenceConfig(
            checkpoint_path=wav2lip_path,
            face=reference_avatar_path,
            audio="temp/t2s_output.wav",
            outfile="temp/synced_video.mp4",
            resize_factor=1
        )

        wav2lip_inference = Wav2LipInference(wav2lip_config)

        
        toggle_output(True)
        print("Hello master, I am your command line private assistant powered by GLM4, GPT-SoVITS and Wav2Lip. What can I do for you today?")
        response = "Hello master, I am your private assistant. What can I do for you today?"
        toggle_output(False)
        synthesize(reference_audio_path, reference_text_path, "英文", response, "英文", temp_file_path, how_to_cut = i18n("按英文句号.切"))
        wav2lip_inference.run_inference()
        toggle_output(True)

        # Launch GUI
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        window.load_video("temp/synced_video.mp4")
        window.append_dialogue(response)

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            response = chatbot.generate_response(user_input)
            # toggle_output(False)
            synthesize(reference_audio_path, reference_text_path, "英文", clean_text(response), "英文", temp_file_path, how_to_cut = i18n("按英文句号.切"))
            wav2lip_inference.run_inference()
            window.load_video("temp/synced_video.mp4")
            window.append_dialogue(response)
            # toggle_output(True)
            print("GLM-4:", response)



    elif language == "zh":
        toggle_output(False)
        # TODO: Load all models works for Chinese
        #  1. GLM-4 2. GPT-SoVITS 3. MuseTalk
        toggle_output(True)
        print("主人您好，我是您由GLM4, GPT-SoVITS and Wav2Lip驱动的的私人助理。请问今天我有什么可以帮到你？")
        response = "主人您好，我是您的的私人助理。请问今天我有什么可以帮到你"
        # Generate the voice and lip movement

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
