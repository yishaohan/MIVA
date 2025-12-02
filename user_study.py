import sys
import csv

import random

import argparse
from omegaconf import OmegaConf

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLabel, QLineEdit, QHBoxLayout, QRadioButton, QButtonGroup, QGridLayout, QProgressBar
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os

class Video(QWidget):
    """
    A QWidget subclass that handles video playback.

    Attributes:
        video_path (str): Path to the video file to be played.
        video_label (QLabel): QLabel widget to display video frames.
        choice_button (QRadioButton): Radio button for user selection.
        capture (cv2.VideoCapture): OpenCV video capture object.
        timer (QTimer): Timer to update video frames at regular intervals.

    Methods:
        update_frame():
            Reads the next frame from the video and updates the display.
        load_video(video_path):
            Loads a new video from the specified path.
        close_video():
            Releases the video capture resource.
    """
    def __init__(self, video_path, window_width=500, window_height=720, parent=None):
        super().__init__(parent)
        self.window_width = window_width
        self.window_height = window_height
        self.video_path = video_path

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.choice_button = QRadioButton(self)

        layout = QVBoxLayout()
        layout.addWidget(self.choice_button, alignment=Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.capture = cv2.VideoCapture(self.video_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_rgb = cv2.resize(frame_rgb, (512, 320))            
        
        image = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def load_video(self, video_path):
        self.capture.release()
        self.video_path = video_path
        self.capture = cv2.VideoCapture(self.video_path)

    def close_video(self):
        self.capture.release()

class WelcomeScreen(QWidget):
    """
    The initial screen of the application.
    Users can enter their name and start the study.

    Attributes:
        data (dict): Dictionary containing categories and associated video/image paths.
            Example: {"cloud-1": ["reference_image_path", "video_1", "video_2", "video_3", "video_4"],
                      "cloud-2": ["reference_image_path", "video_1", "video_2", "video_3", "video_4", "video_5", ...]}
        output_path (str): Path to save the study results (csv_file)
        window_width (int): Width of the main application window.
        window_height (int): Height of the main application window.
        label (QLabel): Displays welcome text.
        name_input (QLineEdit): Input field for the user's name.
        start_button (QPushButton): Button to start the study.

    Methods:
        show_player():
            Initiates the Player widget and closes the welcome screen.
    """

    def __init__(self, data, output_path, window_width, window_height, parent=None):
        super().__init__(parent)
        self.data = data
        self.output_path = output_path
        self.window_width = window_width
        self.window_height = window_height

        text_body = """
        Welcome to the user study.
        You will be shown a series of videos. For each screen, please select the best looking video.
        The study is expected to take around 30 minutes. Your responses will be saved automatically.
        """

        self.label = QLabel(text_body, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("highlight")

        # Add QLineEdit for name input
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter your name")

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.show_player)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

        self.setWindowTitle('Welcome')
        self.resize(800, 500)
        self.show()

    def show_player(self):
        name = self.name_input.text() or "default"
        self.player = Player(self.data, name, self.output_path, self.window_width, self.window_height)
        self.player.show()
        self.close()


class EndScreen(QWidget):
    """
    The final screen displayed at the end of the study.

    Attributes:
        label (QLabel): Displays the thank-you message.
        close_button (QPushButton): Button to end the study and close the window.

    Methods:
        end_study():
            Closes the EndScreen window.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.player = player

        text_body = "\
        Thank you for participating in the user study.\n\
        Your responses have been saved. You can safely close this window.\
        "

        self.label = QLabel(text_body, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.resize(800, 500)

        # # Add "Previous Category" and "Close" buttons
        # self.previous_button = QPushButton("Previous Category", self)
        # self.previous_button.clicked.connect(self.previous_category)
        
        self.close_button = QPushButton("End Study", self)
        self.close_button.clicked.connect(self.end_study)

        # Layout for buttons
        button_layout = QHBoxLayout()
        # button_layout.addWidget(self.previous_button)
        button_layout.addWidget(self.close_button)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    # def previous_category(self):
    #     self.player.current_category_index -= 1
    #     self.player.load_current_category()
    #     self.player.progress_bar.setValue(self.player.current_category_index)
    #     self.player.show()
    #     self.close()

    def end_study(self):
        self.close()


class Player(QWidget):
    """
    The main player widget that displays videos for each category, allows user
    selections, and manages the progression through different categories.

    Attributes:
        data (dict): Dictionary containing categories and associated video/image paths.
            Example: {"cloud-1": ["reference_image_path", "video_1", "video_2", "video_3", "video_4"],
                      "cloud-2": ["reference_image_path", "video_1", "video_2", "video_3", "video_4", "video_5", ...]}
        output_path (str): Path to save the study results.
        window_width (int): Width of the video display window.
        window_height (int): Height of the video display window.
        question_label (QLabel): Label prompting the user to make a selection.
        end_screen (EndScreen): Instance of the EndScreen widget.
        categories (list): list of category names.
        current_category_index (int): Index of the current category being displayed.
        video_paths (list): List of video paths for the current category.
        csv_file (str): Path to the CSV file where results are saved.
        selections (dict): Dictionary storing user selections, will be saved to csv_file once app is closed.
        picture_label (QLabel): Label to display reference image.
        picture_text_label (QLabel): Label for reference image description.
        video_components (list): List of Video widgets for the current category.
        video_text_labels (list): List of labels describing each video.
        next_button (QPushButton): Button to proceed to the next category.
        previous_button (QPushButton): Button to return to the previous category.
        progress_label (QLabel): Label indicating progress.
        progress_bar (QProgressBar): Progress bar showing    study progress.

        n_dim (int): number of dimension to be evaluate
        evaluation_dim (list): a list of String contains the name of the dimension to be evaluate
        n_col (int): number of columns


    Methods:
        resizeEvent(event):
            Overrides the resize event to adjust the reference image size.
        update_picture_size():
            Updates the size of the reference picture based on the window size.
        set_picture(category):
            Sets the reference picture for the given category.
        next_category():
            Moves to the next category or ends the study if at the last category.
        previous_category():
            Moves to the previous category if not at the first category.
        load_current_category():
            Loads the videos and reference image for the current category.
        record_choice(button):
            Records the user's video selection for the current category.
        save_results_to_csv():
            Saves all user selections to a CSV file.
        closeEvent(event):
            Handles the window close event by saving results and releasing resources.
    """

    def __init__(self, data, name, output_path, window_width, window_height, parent=None):
        super().__init__(parent)
        self.question_label = QLabel()
        self.question_label.setObjectName("highlight")
        self.end_screen = None
        
        shuffled_data = shuffle_dict(data)
        self.data = shuffled_data
        self.categories = list(shuffled_data.keys()) 
        self.current_category_index = 0

        self.video_paths = shuffled_data[self.categories[self.current_category_index]][1:]

        ## TODO: Set the evaluation dimensions
        # self.evaluation_dim = ["Overall"]  #### TODO: Be replaced by actual evaluation dimension names (list of String)
        # self.n_dim = len(self.evaluation_dim) ## n is number of dimensions ## TODO: After the actual names are in the list, the n_dim should be te length of the list 
        ## END DO
        
        os.makedirs(output_path, exist_ok=True)
        self.csv_file = f'{output_path}/{name}1_results.csv'

        self.selections = {key: "" for key in list(data.keys())}
        

        # Add a QLabel for displaying reference picture
        self.picture_label = QLabel(self)
        self.picture_label.setAlignment(Qt.AlignCenter)
        self.picture_label.setFixedHeight(320)
        # self.picture_label.setContentsMargins(0, 0, 0, -20)

        self.picture_label.setStyleSheet("margin-top: 19px;")
        # self.picture_label.setScaledContents(True)

        self.picture_text_label = QLabel("     Reference Picture I*", self)

        self.video_components = [Video(path, window_width // 2, window_height // 3) for path in self.video_paths]

        self.video_text_labels = [QLabel(f"   Method {i+1}", self) for i in range(len(self.video_paths))]  
        
        self.choice_group = QButtonGroup(self)
        for i, video in enumerate(self.video_components):
            self.choice_group.addButton(video.choice_button, i)
        
        self.choice_group.buttonClicked.connect(self.record_choice)


        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_category)
        self.previous_button = QPushButton("Previous", self)
        self.previous_button.clicked.connect(self.previous_category)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.previous_button)
        button_layout.addWidget(self.next_button)

        self.grid_layout = QVBoxLayout()
        self.video_grid = QGridLayout()

        self.video_grid.addWidget(self.picture_label, 0, 0)
        self.video_grid.addWidget(self.picture_text_label, 1, 0)

        self.n_col = 4
        for i, video in enumerate(self.video_components):
            row = (i + 1) // self.n_col 
            col = (i + 1) % self.n_col
            self.video_grid.addWidget(video, row * 2, col)
            self.video_grid.addWidget(self.video_text_labels[i], row * 2 + 1, col)


        self.progress_label = QLabel("Progress", self)
        self.progress_label.setAlignment(Qt.AlignLeft)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, len(self.categories) - 1)
        self.progress_bar.setValue(self.current_category_index)
        self.progress_bar.setGeometry(0, 10, 500, 20)

        self.progress_layout = QHBoxLayout()
        self.progress_layout.addWidget(self.progress_label)
        self.progress_layout.addWidget(self.progress_bar)

        # Add the grid and other elements to the layout
        self.grid_layout.addWidget(self.question_label)
        self.grid_layout.addLayout(self.video_grid)
        self.grid_layout.addLayout(self.progress_layout)
        self.grid_layout.addLayout(button_layout)

        self.setLayout(self.grid_layout)

        self.setWindowTitle('Video Player')
        self.move(1,1)

        self.load_current_category()

    # Override resizeEvent to fix image sizing issue
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_picture_size()


    def update_picture_size(self):
        self.set_picture(self.categories[self.current_category_index])


    def set_picture(self, category):
        picture_path = self.data[category][0]
        pixmap = QPixmap(picture_path)
        scaled_pixmap = pixmap.scaled(self.picture_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.picture_label.setPixmap(scaled_pixmap)


    def next_category(self):
        if len(self.selections[self.categories[self.current_category_index]]) < 1:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Alert!")
            dlg.setText("Please select the best video to go to the next page!")
            dlg.exec()

            return
        if self.current_category_index < len(self.categories) - 1 :
            self.current_category_index += 1
            self.load_current_category()
            self.progress_bar.setValue(self.current_category_index)
        else:
            # Show EndScreen when the last category is reached
            self.hide()
            self.save_results_to_csv()  
            self.end_screen = EndScreen()
            self.end_screen.show()


    def previous_category(self):
        if self.current_category_index > 0:
            self.current_category_index -= 1
            self.load_current_category()
            self.progress_bar.setValue(self.current_category_index)        


    def load_current_category(self):
        current_category = self.categories[self.current_category_index]
        category_label = current_category.split("-")[0]
        self.question_label.setText(f"Select the best looking video: {category_label}")

        # Set picture
        self.set_picture(current_category)

        # Load the videos for the current category
        video_paths = self.data[current_category][1:]
        for i, video_path in enumerate(video_paths):
            self.video_components[i].load_video(video_path)

        # Uncheck all radio buttons
        self.choice_group.setExclusive(False)
        for button in self.choice_group.buttons():
            button.setChecked(False)
        self.choice_group.setExclusive(True)

        # Load previous selection if exists
        if current_category in self.selections:
            selected_video_path = self.selections[current_category]
            if selected_video_path in video_paths:
                selected_index = video_paths.index(selected_video_path)
                self.choice_group.button(selected_index).setChecked(True)


    def record_choice(self, button):
        selected_index = self.choice_group.id(button)
        current_category = self.categories[self.current_category_index]
        selected_video_path = self.data[current_category][selected_index + 1]  # Account for picture being at index 0
        self.selections[current_category] = selected_video_path

    def save_results_to_csv(self):
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for category, video_path in self.selections.items():
                if video_path != "":
                    video_path = video_path.split('/')[-3] ## get the generation metod by get the directory folder name
                writer.writerow([category, video_path])

    def closeEvent(self, event):
        self.save_results_to_csv()
        for video in self.video_components:
            video.close_video()
        event.accept()

def shuffle_dict(data):
    keys_list = list(data)
    random.shuffle(keys_list)
    shuffled_data = {}
    for key in keys_list:
        shuffled_data[key] = data[key]
    return shuffled_data

def main(
    video_paths: list[str],
    image_path: str,
    categories: dict,
    output_path: str,
    window_width: int = 1280,
    window_height: int = 960
):
    app = QApplication(sys.argv)
    app.setStyleSheet("QLabel {font-size: 10pt;} QLabel#highlight {font-size: 15pt;}")
    

    ## TODO: select the range of the data from the config.yaml file
    data = {}
    for category in categories: ## category is the key of the dictionary
        if categories[category] == "all":
            for i in range(1, 21): #####     
                # Data folder are assumed to contain 1.mp4 -> 20.mp4
                key = f"{str(category)}-{str(i)}"
                value = [f"{image_path}/{str(category)}/{i}.png"]
                for path in video_paths:
                    value.append(f"{path}/{str(category)}/{i}.mp4")
                ## video shuffle
                value_video = value[1:]
                random.shuffle(value_video)
                value[1:] = value_video
                
                data[key] = value
        elif type(categories[category]) is list:
            video_to_test = categories[category]
            for i in video_to_test:
                if i in range(1, 21):
                    key = f"{str(category)}-{str(i)}"
                    value = [f"{image_path}/{str(category)}/{i}.png"]
                    for path in video_paths:
                        value.append(f"{path}/{str(category)}/{i}.mp4")
                    ## video shuffle
                    value_video = value[1:]
                    random.shuffle(value_video)
                    value[1:] = value_video
                    
                    data[key] = value
        elif categories[category][:6] == "random":
            n_rand = int(categories[category][7:])
            rand_sample = random.sample(range(1, 21), n_rand)
            for i in rand_sample:
                key = f"{str(category)}-{str(i)}"
                value = [f"{image_path}/{str(category)}/{i}.png"]
                for path in video_paths:
                    value.append(f"{path}/{str(category)}/{i}.mp4")
                ## video shuffle
                value_video = value[1:]
                random.shuffle(value_video)
                value[1:] = value_video

                data[key] = value

    ## END DO


    welcome_screen = WelcomeScreen(data, output_path, window_width, window_height)
    sys.exit(app.exec_())


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(**config)