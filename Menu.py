
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

#Refereneces\en\Yae_Miko_V2_Genshin5.1.txt
#References\en\Yae_Miko_V2_Genshin5.1.txt
# print(os.path.exists("References\\en\\Yae_Miko_V2_Genshin5.1.txt"))

import re

def split_and_group_sentences(text):
    # Split the text into sentences based on `.`, `?`, or `!` followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Group every two sentences together
    grouped_sentences = [' '.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
    return grouped_sentences

# Your input string
text = '''Certainly! "Orion" can refer to several different things, including a constellation in the night sky, a spacecraft, and other celestial objects or concepts. Here are some details related to these aspects: ### Constellation: **Name:** Orion is one of the most prominent constellations in the night sky. **Location:** It\'s located on the celestial equator and is visible year-round from many parts of Earth. **Stars:** - **Rigel**: The brightest star in Orion. - **Betelgeuse**: Also known as Alpha Orionis, it has a reddish hue. - **Bellatrix**: Known for its blue color, it represents Orion\'s shoulder. The pattern forms an approximate quadrilateral shape with four distinct stars at each corner. There are also famous regions within this constellation such as the Orion Nebula (M42), which is a great star-forming region, and the Horsehead Nebula, another well-known feature. ### Spacecraft: **NASA\'s Orion Program:** This program involves the development of NASA\'s next generation human spaceflight system designed to carry astronauts beyond low-Earth orbit and back. The spacecraft itself is named after the constellation and is intended to be reusable and capable of landing on the Moon, Mars, and asteroids. Key elements include: - **Crew Module**: Designed for carrying astronauts and their life support systems. - **Service Module**: Provides propulsion, power, thermal control, and storage for cargo. - **Exploration Stage**: A solar-powered upper stage that provides additional propulsive capability. Orion was first launched into orbit by the Space Launch System (SLS) rocket during Exploration Mission 1 (EM-1) in December 2022, marking the beginning of Artemis missions aimed at returning humans to the lunar surface. ### Other Celestial Objects: There are numerous other objects associated with the name "Orion": - **Orion Planetary Society**: An organization dedicated to promoting planetary exploration and education. - **Orion Key**: A type of key used in lockpicking; not directly related to the constellation but often confused due to the similarity in names. - **Orion\'s Belt**: This refers to three bright stars forming a distinctive belt-like pattern in the constellation. They represent Orion’s belt in mythology. Each of these references highlights how the name "Orion" touches upon various areas of astronomy, culture, and technology.'''

# Split and group sentences
grouped_sentences = split_and_group_sentences(text)

# Print the result
# for i, group in enumerate(grouped_sentences, 1):
#     print(f"Group {i}: {group}\n")

print(grouped_sentences)