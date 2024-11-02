import sys
import os
import contextlib

from chatglm.glm4_module import GLMChatbot

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





if __name__ == "__main__":
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
        print("GLM-4:", response)
