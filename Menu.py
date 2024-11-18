
import time

def print_with_delay(text, delay=0.3):
    """
    Print text word by word with a delay.

    Args:
        text (str): The text to print.
        delay (float): Delay in seconds between each word. Default is 0.3 seconds.
    """
    words = text.split()  # Split the text into words
    for word in words:
        print(word, end=' ', flush=True)
        time.sleep(delay)
    print()  # Print a newline at the end

# Example usage
language_text = {
    "English": "You selected English. Now the program will continue in English.",
    "Chinese": "您选择了中文。现在程序将继续使用中文。",
}

selected_language = "English"  # Simulating user selection
print_with_delay(language_text[selected_language], delay=0.3)
