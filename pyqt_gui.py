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
    QSlider,
    QLabel,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtCore import QUrl, QSizeF, Qt, pyqtSignal
from PyQt5.QtGui import QFont


class MainWindow(QMainWindow):

    user_input_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player and Dialogue Box")
        self.resize(800, 1600)

        # Main layout
        main_layout = QVBoxLayout()

        # Video Player
        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        main_layout.addWidget(self.graphics_view)

        progress_layout = QHBoxLayout()

        # prograss bar
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0)  # Range will be set dynamically
        self.progress_slider.sliderMoved.connect(self.seek_video)  # Seek when user drags
        progress_layout.addWidget(self.progress_slider)

        self.time_label = QLabel("00:00 / 00:00")
        progress_layout.addWidget(self.time_label)

        main_layout.addLayout(progress_layout)

        control_layout = QHBoxLayout()
        # Pause/Play Button
        self.pause_play_button = QPushButton("Pause")
        control_layout.addWidget(self.pause_play_button)
        main_layout.addLayout(control_layout)

        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_item)
        self.stop_button = QPushButton("Stop Video")
        main_layout.addWidget(self.stop_button)
        self.stop_button.clicked.connect(self.media_player.stop)

        # Dialogue Box
        self.dialogue_box = QTextBrowser()
        main_layout.addWidget(self.dialogue_box)

        # User Input and Submit Button
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.submit_button = QPushButton("Submit")
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.submit_button)
        main_layout.addLayout(input_layout)

        

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set the initial size of the video item
        self.update_video_size()

        # Connect submit button
        self.submit_button.clicked.connect(self.emit_user_input)
        # self.submit_button.clicked.connect(self.get_user_input)
        self.pause_play_button.clicked.connect(self.toggle_pause_play)

        # Connect media player signals
        self.media_player.durationChanged.connect(self.update_slider_range)
        self.media_player.positionChanged.connect(self.update_slider_position)

    def toggle_pause_play(self):
        """Toggle between playing and pausing the video."""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.pause_play_button.setText("Play")
        else:
            self.media_player.play()
            self.pause_play_button.setText("Pause")

    def resizeEvent(self, event):
        """Handle window resize and adjust video size."""
        super().resizeEvent(event)
        self.update_video_size()

    def update_video_size(self):
        """Update the video size to fit the graphics view."""
        size = self.graphics_view.size()
        # self.video_item.setSize(QSizeF(size.width(), size.height()))
        self.video_item.setSize(QSizeF(500, 800))

    def load_video(self, video_path):
        """Load and play video from the given path."""
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.media_player.play()

    def append_dialogue(self, text):
        """Append text to the dialogue box."""
        self.dialogue_box.append(text)

    def get_user_input(self):
        """Handle user input when submit is clicked."""
        user_text = self.user_input.text()
        # print(f"User input: {user_text}")  # You can process it as needed
        self.user_input.clear()
        return user_text


    def update_slider_range(self, duration):
        """Set the slider range when the video duration is known."""
        self.progress_slider.setRange(0, duration)
        self.update_time_label(0, duration)  # Update label initially

    def update_slider_position(self, position):
        """Update the slider's position as the video plays."""
        self.progress_slider.setValue(position)
        self.update_time_label(position, self.media_player.duration())

    def seek_video(self, position):
        """Seek the video to the specified position when the slider is moved."""
        self.media_player.setPosition(position)

    def update_time_label(self, current_position, total_duration):
        """Update the time label with the current and total time."""
        current_time = self.format_time(current_position)
        total_time = self.format_time(total_duration)
        self.time_label.setText(f"{current_time} / {total_time}")

    @staticmethod
    def format_time(milliseconds):
        """Format time from milliseconds to mm:ss."""
        seconds = milliseconds // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"
    
    def emit_user_input(self):
        """Emit the user input via the custom signal."""
        user_text = self.user_input.text()
        self.user_input_signal.emit(user_text)  # Emit the input
        self.user_input.clear()  # Clear the input field


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial"))
    window = MainWindow()

    def handle_user_input(text):
        print(f"Received user input: {text}")
        window.dialogue_box.append(f"User said: {text}")
        return text

    window.user_input_signal.connect(handle_user_input)
    
    window.show()

    # Example usage:
    window.load_video("temp/synced_video.mp4")
    for i in range(0,10):
        window.append_dialogue("Hello, how are you?\n\nHoooo")

    sys.exit(app.exec_())