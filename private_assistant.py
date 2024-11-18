import sys
import os
import contextlib
<<<<<<< Updated upstream

# from chatglm.glm4_module import GLMChatbot
=======
import re
from chatglm.glm4_module import GLMChatbot
>>>>>>> Stashed changes

_stdout_backup = sys.stdout
_stderr_backup = sys.stderr


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

def clean_text(text):
    # Replace newline and escaped single quotes
    cleaned_text = text.replace("\\n", "").replace("\\'", "'").replace("\\t", "")
    # Optionally, replace multiple spaces with a single space for better formatting
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

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



def main():
    # This is the folder path that temporary store wav file and video file.
    temp_file_path = "temp"
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)
        print(f"Folder created: {temp_file_path}")

    models_parent_path = "Models_Pretrained"
    museTalk_model_path = os.path.join(models_parent_path, "MuseTalk")
    wav2lip_path = os.path.join(models_parent_path, "Wav2Lip")
    


    print("Please select your language:")
    print("1. English")
    print("2. Chinese")



    language = "en"


    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice == "1":
            language = "en"
            print("You selected English.")
            break
        elif choice == "2":
            language = "zh"
            print("You selected Chinese.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")





    if language == "en":
        
        toggle_output(False)
        # TODO: Load all models works for English
        #  1. GLM-4 2. GPT-SoVITS 3. Wav2Lip
        toggle_output(True)
        print("Hello master, I am your command line private assistant powered by GLM4, GPT-SoVITS and Wav2Lip. What can I do for you today?")
        response = "Hello master, I am your command line private assistant. What can I do for you today?"
        # Generate the voice and lip movement
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            # response = chatbot.generate_response(user_input)
            print("GLM-4:", response)



    elif language == "zh":
        toggle_output(False)
        # TODO: Load all models works for Chinese
        #  1. GLM-4 2. GPT-SoVITS 3. MuseTalk
        toggle_output(True)
        print("主人您好，我是您由GLM4, GPT-SoVITS and Wav2Lip驱动的的私人助理。请问今天我有什么可以帮到你？")
        response = "主人您好，我是您的的私人助理。请问今天我有什么可以帮到你"
        # Generate the voice and lip movement









    # toggle_output(False)
    # chatbot = GLMChatbot()

    # # Begin interaction
    # toggle_output(True)
    # print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")
    # while True:
    #     user_input = input("\nYou: ")
    #     if user_input.lower() in ["exit", "quit"]:
    #         break
    #     response = chatbot.generate_response(user_input)
    #     print("GLM-4:", response)



if __name__ == "__main__":
<<<<<<< Updated upstream
    main()
=======
    toggle_output(False)
    chatbot = GLMChatbot()

    # Begin interaction
    toggle_output(True)
    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot.generate_response(user_input)
        response = clean_text(remove_emojis(response))
        print("\nGLM-4:", response)
>>>>>>> Stashed changes
