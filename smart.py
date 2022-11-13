import cv2
from cv2 import Mat
import dlib
import math
import pyautogui
import mouse
import pyttsx3
from PyQt5.QtWidgets import (
    QApplication, 
    QPushButton, 
    QWidget, 
    QVBoxLayout, 
    QLabel, 
    QMainWindow, 
    QMenuBar, 
    QGridLayout,
    QCheckBox
)
from PyQt5.QtGui import  QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, Qt
from PyQt5 import QtGui

import numpy as np
import keyboard
import pyqtgraph as pg
import json

BLINK_RATIO_THRESHOLD: float = 5.7
pyautogui.FAILSAFE = False

# active = True
# spam_mode = False


# def set_mode():
#     global spam_mode
#     spam_mode = not spam_mode
#     print(spam_mode)
#     pyttsx3.speak('Spam mode is on. Press ctrl+shift+e to turn clicker off' if spam_mode else 'Spam mode is off. Press ctrl+shift+e to turn clicker on')

# def set_is_on():
#     global active
#     active = not active
#     pyttsx3.speak('Clicker is active. Press ctrl+shift+o to turn clicker off' if active else 'Clicker is inactive. Press ctrl+shift+o to turn clicker on')

with open('settings.json', 'r') as f:
    settings: dict[str, bool] = json.load(f)

def dump_settings():
    controller.terminate()

    with open('settings.json', 'w') as f:
        json.dump(settings,f)


def convert_cv_qt(cv_img: Mat):
    """Convert from an opencv image to QPixmap"""
    rgb_image: Mat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line: int = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)  # type: ignore
    p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)  # type: ignore
    return QPixmap.fromImage(p)


app: QApplication = QApplication([])

app.setApplicationName('Blink Launcher')
app.setApplicationVersion('1.0.0')
app.setOrganizationName('Coders Perks .Inc')



window: QWidget = QWidget()
window.resize(800,600)


webcam = QLabel(parent=window)
webcam.setMinimumSize(300,400)

@pyqtSlot(np.ndarray)
def update_image(cv_img: np.ndarray) -> None:
    qt_img = convert_cv_qt(cv_img)
    webcam.setPixmap(qt_img)



class GraphController(QWidget):
    
    def __init__(self, parent) -> None:
        super().__init__(parent)  # type: ignore
        self.graph = pg.PlotWidget(self)
        self.data_x = [0]
        self.data_y = [0]

        self.data_line = self.graph.plot(self.data_x,self.data_y)
        self.graph.show()


    @pyqtSlot(int)
    def set_blink_ratio(self,blink_ratio: int) -> None:
        self.blink_ratio = blink_ratio
        self.update_plot_data(self.blink_ratio,self.data_line,self.data_x, self.data_y)
    
    def update_plot_data(self,n: int,data_line,x: list[int],y: list[int]) -> None:

        x.append(x[-1] + 1)  # Add a new value 1 higher than the last.
        y.append(n)

        data_line.setData(x, y) 



layout = QVBoxLayout()

w = QMainWindow()
w.setCentralWidget(window)

menu_bar = QMenuBar(window)
file_menu = menu_bar.addMenu('&File')
edit_menu = menu_bar.addMenu('&Edit')

preferences = edit_menu.addAction('&Preferences')
exit_action = file_menu.addAction('&Exit')

prefer_window = 0


def preferences_window():
    global prefer_window
    
    prefer_window = QMainWindow()
    window_ = QWidget()
    prefer_window.setWindowTitle('Preferences')
    layout_a = QGridLayout()
    window_.setLayout(layout_a)

    buttons: list[QCheckBox] = []

    def button_clicked():
        for button in buttons:
            if button.isChecked():
                settings[button.setting_name] = not settings[button.setting_name]
                print(settings[button.setting_name])


    i = 0
    for setting, value in settings.items():
        print(f'{setting}:{value}')
        label = QLabel(setting, parent=window_)
        
        layout_a.addWidget(label,i, 0)
        button = QCheckBox(parent=window_)
        

        layout_a.addWidget(button, i, 1)
        
        button.setting_name = setting
        button.setChecked(settings[button.setting_name])

        buttons.append(button)
        button.clicked.connect(button_clicked)
        
        label.show()
        button.show()
        i += 1

    apply_button = QPushButton('&Apply')
    prefer_window.setCentralWidget(window_)
    prefer_window.show()

def exit_app():
    app.exit(0)

preferences.triggered.connect(preferences_window)
exit_action.triggered.connect(exit_app)  # type: ignore


w.setMenuBar(menu_bar)
w.show()

button: QPushButton = QPushButton('Start Blink Clicker',window)
button.setMaximumWidth(200)
button.show()

number_of_blinks = QLabel(window)
number_of_blinks.setText('Number of Blinks: N/A')
number_of_blinks.show()

layout.addWidget(number_of_blinks)
layout.addWidget(button)
layout.addWidget(webcam)
window.setLayout(layout)

window.show()



def start_button():
    controller.start()
class VideoController(QThread):
    signal_pixel_map = pyqtSignal(np.ndarray)
    blink_ratio_sender = pyqtSignal(int)

    def __init__(self, parent, headless) -> None:
        super().__init__(parent)
        self.headless = headless
        self.g = GraphController(window)

        layout.addWidget(self.g)


    def run(self):
        self.blink_detection()
    def blink_detection(self):

        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Blink Clicker')
        cv2.resizeWindow('Blink Clicker', 1,1)
        detector = dlib.get_frontal_face_detector()  # type: ignore
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # type: ignore

        #these landmarks are based on this image(https://miro.medium.com/max/600/1*PjBKmgNHtC3dRE7ZRSD7qA.png)
        left_eye_landmarks = [36, 37, 38, 39, 40, 41]
        right_eye_landmarks = [42, 43, 44, 45, 46, 47]

        def midpoint(point1 ,point2):
            return int((point1.x + point2.x)/2), int((point1.y + point2.y)/2)

        def euclidean_distance(point1 , point2):
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)



        def get_blink_ratio(eye_points, facial_landmarks):

            #loading all the required points
            corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                            facial_landmarks.part(eye_points[0]).y)
            corner_right = (facial_landmarks.part(eye_points[3]).x, 
                            facial_landmarks.part(eye_points[3]).y)
            
            center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                                    facial_landmarks.part(eye_points[2]))
            center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 

                                    facial_landmarks.part(eye_points[4]))
            #calculating distance
            horizontal_length = euclidean_distance(corner_left,corner_right)
            vertical_length = euclidean_distance(center_top,center_bottom)

            cv2.line(frame,corner_left,corner_right,(255,0,0))
            cv2.line(frame,center_top,center_bottom,(255,0,0))

            for points in right_eye_landmarks:
                cv2.circle(frame,(landmarks.part(points).x, landmarks.part(points).y),2,(255,0,0))

            for points in left_eye_landmarks:
                cv2.circle(frame,(landmarks.part(points).x, landmarks.part(points).y),2,(255,0,0))



            ratio = horizontal_length / vertical_length
            return ratio

        is_changed = lambda prev_pos, pos: True if prev_pos != pos else False
        # pyttsx3.speak('Spam mode is on. Press ctrl+shift+e to turn clicker off' if spam_mode else 'Spam mode is off. Press ctrl+shift+e to turn clicker on')
        # pyttsx3.speak('Clicker is active. Press ctrl+shift+o to turn clicker off' if spam_mode else 'Clicker is inactive. Press ctrl+shift+o to turn clicker on')

        n_blinks = 0
        while True:
            #capturing frame
            retval, frame = cap.read()
            
            #exit the application if frame not found
            if not retval:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
            faces,_,_ = detector.run(image = frame, upsample_num_times = 0,
                                    adjust_threshold = 0.0)

            for face in faces:
                print('ok')
                landmarks = predictor(frame, face)
                point = (landmarks.part(36).x, landmarks.part(36).y)
                left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
                right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
                blink_ratio   = (left_eye_ratio + right_eye_ratio) / 2

                print(blink_ratio)


                if blink_ratio > BLINK_RATIO_THRESHOLD:
                    n_blinks += 1
                    # pyttsx3.speak('Blink detected')
                    #Blink detected! Do Something!
                    number_of_blinks.setText(f'Number of blinks {n_blinks}')
                    cv2.putText(frame,"BLINKING",(10,50), cv2.FONT_HERSHEY_SIMPLEX,
                                2,(255,255,255),2,cv2.LINE_AA)
         
                    for i in range(10):
                        pyautogui.click(pyautogui.position())             
                        mouse.click('left')
                        # else:
                        #     pyautogui.click(pyautogui.position())
                        #     mouse.click('left')
                    # else:
                    #     continue
                # self.blink_ratio_sender.emit(blink_ratio)

                self.g.set_blink_ratio(blink_ratio)                        

            if self.headless == False:
                cv2.imshow('Blink Clicker', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            if keyboard.is_pressed(27):
                break
            self.signal_pixel_map.emit(frame)


        #releasing the VideoCapture object
        cap.release()
        cv2.destroyAllWindows()

controller = VideoController(window, settings['headless'])
controller.signal_pixel_map.connect(update_image)  # type: ignore
button.clicked.connect(start_button)

app.aboutToQuit.connect(dump_settings)

print('about to start')

app.exec()