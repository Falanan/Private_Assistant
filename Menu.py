
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

# import re

# def split_and_group_sentences(text):
#     # Split the text into sentences based on `.`, `?`, or `!` followed by whitespace
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     # Group every two sentences together
#     grouped_sentences = [' '.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
#     return grouped_sentences

# # Your input string
# text = '''Certainly! "Orion" can refer to several different things, including a constellation in the night sky, a spacecraft, and other celestial objects or concepts. Here are some details related to these aspects: ### Constellation: **Name:** Orion is one of the most prominent constellations in the night sky. **Location:** It\'s located on the celestial equator and is visible year-round from many parts of Earth. **Stars:** - **Rigel**: The brightest star in Orion. - **Betelgeuse**: Also known as Alpha Orionis, it has a reddish hue. - **Bellatrix**: Known for its blue color, it represents Orion\'s shoulder. The pattern forms an approximate quadrilateral shape with four distinct stars at each corner. There are also famous regions within this constellation such as the Orion Nebula (M42), which is a great star-forming region, and the Horsehead Nebula, another well-known feature. ### Spacecraft: **NASA\'s Orion Program:** This program involves the development of NASA\'s next generation human spaceflight system designed to carry astronauts beyond low-Earth orbit and back. The spacecraft itself is named after the constellation and is intended to be reusable and capable of landing on the Moon, Mars, and asteroids. Key elements include: - **Crew Module**: Designed for carrying astronauts and their life support systems. - **Service Module**: Provides propulsion, power, thermal control, and storage for cargo. - **Exploration Stage**: A solar-powered upper stage that provides additional propulsive capability. Orion was first launched into orbit by the Space Launch System (SLS) rocket during Exploration Mission 1 (EM-1) in December 2022, marking the beginning of Artemis missions aimed at returning humans to the lunar surface. ### Other Celestial Objects: There are numerous other objects associated with the name "Orion": - **Orion Planetary Society**: An organization dedicated to promoting planetary exploration and education. - **Orion Key**: A type of key used in lockpicking; not directly related to the constellation but often confused due to the similarity in names. - **Orion\'s Belt**: This refers to three bright stars forming a distinctive belt-like pattern in the constellation. They represent Orion’s belt in mythology. Each of these references highlights how the name "Orion" touches upon various areas of astronomy, culture, and technology.'''

# # Split and group sentences
# grouped_sentences = split_and_group_sentences(text)

# # Print the result
# # for i, group in enumerate(grouped_sentences, 1):
# #     print(f"Group {i}: {group}\n")

# print(grouped_sentences)

# import sys
# from PyQt5.QtWidgets import (
#     QApplication,
#     QMainWindow,
#     QVBoxLayout,
#     QHBoxLayout,
#     QWidget,
#     QPushButton,
#     QLineEdit,
#     QTextBrowser,
# )
# from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
# from PyQt5.QtMultimediaWidgets import QVideoWidget
# from PyQt5.QtCore import QUrl
# from PyQt5.QtGui import QFont
# from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
# from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("Video Player and Dialogue Box")
#         self.resize(800, 600)

#         # Main layout
#         main_layout = QVBoxLayout()

#         # Video Player
#         # self.video_player = QVideoWidget()
#         # main_layout.addWidget(self.video_player)
#         self.scene = QGraphicsScene()
#         self.graphics_view = QGraphicsView(self.scene)
#         self.video_item = QGraphicsVideoItem()
#         self.scene.addItem(self.video_item)
#         main_layout.addWidget(self.graphics_view)

#         # Dialogue Box
#         self.dialogue_box = QTextBrowser()
#         main_layout.addWidget(self.dialogue_box)

#         # User Input and Submit Button
#         input_layout = QHBoxLayout()
#         self.user_input = QLineEdit()
#         self.submit_button = QPushButton("Submit")
#         input_layout.addWidget(self.user_input)
#         input_layout.addWidget(self.submit_button)
#         main_layout.addLayout(input_layout)

#         # Central widget
#         central_widget = QWidget()
#         central_widget.setLayout(main_layout)
#         self.setCentralWidget(central_widget)

#         # Media Player
#         self.media_player = QMediaPlayer()
#         # self.media_player.setVideoOutput(self.video_player)
#         self.media_player.setVideoOutput(self.video_item)

#         # Connect submit button
#         self.submit_button.clicked.connect(self.get_user_input)

#     def load_video(self, video_path):
#         """Load and play video from the given path."""
#         self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
#         self.media_player.play()

#     def append_dialogue(self, text):
#         """Append text to the dialogue box."""
#         self.dialogue_box.append(text)

#     def get_user_input(self):
#         """Handle user input when submit is clicked."""
#         user_text = self.user_input.text()
#         print(f"User input: {user_text}")  # You can process it as needed
#         self.user_input.clear()


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     app.setFont(QFont("Arial")) 
#     window = MainWindow()
#     window.show()

#     # Example usage:
#     window.load_video("temp/synced_video.mp4")
#     window.append_dialogue("Hello, how are you?")

#     sys.exit(app.exec_())



import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLineEdit,
    QTextBrowser,
    QGraphicsView,
    QGraphicsScene,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtCore import QUrl, QSizeF
from PyQt5.QtGui import QFont


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player and Dialogue Box")
        self.resize(800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Video Player
        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        main_layout.addWidget(self.graphics_view)

        # Dialogue Box
        self.dialogue_box = QTextBrowser()
        main_layout.addWidget(self.dialogue_box)

        # User Input and Control Buttons
        control_layout = QHBoxLayout()

        # User Input
        self.user_input = QLineEdit()
        control_layout.addWidget(self.user_input)

        # Submit Button
        self.submit_button = QPushButton("Submit")
        control_layout.addWidget(self.submit_button)

        # Pause/Play Button
        self.pause_play_button = QPushButton("Pause")
        control_layout.addWidget(self.pause_play_button)

        main_layout.addLayout(control_layout)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Media Player
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_item)

        # Set the initial size of the video item
        self.set_video_size(640, 360)

        # Connect buttons
        self.submit_button.clicked.connect(self.get_user_input)
        self.pause_play_button.clicked.connect(self.toggle_pause_play)

    def set_video_size(self, width, height):
        """Set the video size manually."""
        self.video_item.setSize(QSizeF(width, height))

    def load_video(self, video_path):
        """Load and play video from the given path."""
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.media_player.play()
        self.pause_play_button.setText("Pause")  # Reset button text

    def toggle_pause_play(self):
        """Toggle between playing and pausing the video."""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.pause_play_button.setText("Play")
        else:
            self.media_player.play()
            self.pause_play_button.setText("Pause")

    def append_dialogue(self, text):
        """Append text to the dialogue box."""
        self.dialogue_box.append(text)

    def get_user_input(self):
        """Handle user input when submit is clicked."""
        user_text = self.user_input.text()
        print(f"User input: {user_text}")  # You can process it as needed
        self.user_input.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial"))
    window = MainWindow()
    window.show()

    # Example usage:
    window.load_video("temp/synced_video.mp4")
    window.append_dialogue("Hello, how are you?")

    sys.exit(app.exec_())
