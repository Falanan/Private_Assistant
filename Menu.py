
# import time

# def print_with_delay(text, delay=0.3):
#     """
#     Print text word by word with a delay.

#     Args:
#         text (str): The text to print.
#         delay (float): Delay in seconds between each word. Default is 0.3 seconds.
#     """
#     words = text.split()  # Split the text into words
#     for word in words:
#         print(word, end=' ', flush=True)
#         time.sleep(delay)
#     print()  # Print a newline at the end

# # Example usage
# language_text = {
#     "English": "You selected English. Now the program will continue in English.",
#     "Chinese": "您选择了中文。现在程序将继续使用中文。",
# }

# selected_language = "English"  # Simulating user selection
# print_with_delay(language_text[selected_language], delay=0.3)



import os

# def ensure_path_exists(path):
#     """
#     Ensure the folder path exists. If it doesn't, create it.

#     Args:
#         path (str): The path to check and create if necessary.
#     """
#     folder_path = os.path.dirname(path) if os.path.isfile(path) else path

#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#         print(f"Path created: {folder_path}")
#     else:
#         print(f"Path already exists: {folder_path}")

# # Example usage
# # Provide a folder path
# ensure_path_exists("path/to/folder")

# # Provide a file path
# ensure_path_exists("path/to/folder/file.txt")



def list_files_in_folder(folder_path):
    """
    List all files in a given folder using os.scandir().

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of file names in the folder.
    """
    try:
        with os.scandir(folder_path) as entries:
            files = [entry.name for entry in entries if entry.is_file()]
        return files
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return []

# Example usage
folder_path = "Models_Pretrained\GPT_SoVITS\en\GPT_weights_v2"
files = list_files_in_folder(folder_path)
print("Files:", files)


