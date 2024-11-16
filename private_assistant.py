import sys
import os
import contextlib

# from chatglm.glm4_module import GLMChatbot

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


def main():
    print("Please select your language:")
    print("1. English")
    print("2. Chinese")

    choice = "0"

    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice == "1":
            print("You selected English.")
            break
        elif choice == "2":
            print("You selected Chinese.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print("User input: ", choice)

    if choice == "1":
        
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
            response = chatbot.generate_response(user_input)
            print("GLM-4:", response)



    elif choice == "2":
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
    main()