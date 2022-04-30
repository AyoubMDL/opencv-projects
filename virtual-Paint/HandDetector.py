import cv2
import mediapipe as mp
import numpy as np
from utils import *
import time
import math

frame_width = 720
frame_height = 480
brightness = 10

eraser_position = 48
black_position = 135
red_position = 218
blue_position = 307
green_position = 401

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectConf=0.5, trackConf=0.5):
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectConf = detectConf
        self.trackConf = trackConf
        self.circleSelected = False
        self.points = []
        self.canDraw = False

        self.select_activated = False
        self.draw_activated = False

        self.mpHands = mp.solutions.hands
        self.handDetect = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def find_hand(self, image, draw=True):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.handDetect.process(rgb_image)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return image

    def find_position(self, image):
        positions = []
        reel_position = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    img_height, img_width, img_depth = image.shape
                    cx, cy = int(img_width * landmark.x), int(img_height * landmark.y)
                    reel_position.append([idx, landmark.x, landmark.y])
                    """
                    if idx == hand_index:
                        # print(idx, cx, cy)
                        pass
                    #    self.points.append([cx, cy])
                    """
                    positions.append([idx, cx, cy])
        return positions, reel_position

    def get_finger_position(self, image, finger_index):
        positions, _ = self.find_position(image)
        return positions[finger_index][1], positions[finger_index][2]

    def compute_distance_between_two_fingers(self, image, first_idx, second_idx):
        distance = 0
        positions, reel_position = self.find_position(image)
        if len(reel_position) != 0:
            distance = math.sqrt((reel_position[first_idx][1] - reel_position[second_idx][1]) ** 2 + (
                    reel_position[first_idx][2] - reel_position[second_idx][2]) ** 2)
            cv2.line(image, (positions[first_idx][1], positions[first_idx][2]),
                     (positions[second_idx][1], positions[second_idx][2]),
                     (255, 0, 255), 3)
            cv2.putText(image, str(round(distance, 2)),
                        ((positions[first_idx][1] + positions[second_idx][1]) // 2,
                         ((positions[first_idx][2] + positions[second_idx][2]) // 2) - 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return distance

    def index_up_and_middle_finger_down(self, image):
        return self.compute_distance_between_two_fingers(image, 8, 7) > 0.05 > self.compute_distance_between_two_fingers(image, 12, 11)

    def index_and_middle_finger_up(self, image):
        return self.compute_distance_between_two_fingers(image, 8, 7) > 0.05 and self.compute_distance_between_two_fingers(image, 12, 11) > 0.06

    def choice_state(self, image):
        index_positions = self.get_finger_position(image, 8)
        middle_positions = self.get_finger_position(image, 12)
        if self.index_up_and_middle_finger_down(image):
            self.draw_activated = True
            self.select_activated = False
            cv2.circle(image, index_positions, 20, (255, 0, 0), cv2.FILLED)
        elif self.index_and_middle_finger_up(image):
            self.draw_activated = False
            self.select_activated = True
            cv2.circle(image, middle_positions, 20, (0, 0, 0), cv2.FILLED)
        else:
            self.draw_activated = False
            self.select_activated = False
            pass

    def select_choice(self, x=0, y=0):
        path = "normal.png"
        if x < 125 and self.select_activated:
            if y <= eraser_position:
                path = "eraser.png"
            elif eraser_position <= y <= black_position:
                path = "black.png"
            elif red_position <= y <= blue_position:
                path = "blue.png"
            elif black_position <= y <= red_position:
                path = "red.png"
            elif  blue_position <= y <= green_position:
                path = "green.png"

        return path



    """
    def selectCircle(self, img):
        # 370 <= myPos[8][1] <= 430 and 40 <= myPos[8][2] <= 100
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        # myPos, reelPos = self.findPosition(img)
        # if len(myPos) != 0:
        if key == ord('d'):
            self.canDraw = True
            # self.circleSelected = True
        if key == ord('s'):
            self.canDraw = False

        if key == ord('c'):
            self.clearCanvas()
            # self.circleSelected = False
        if self.canDraw:
            self.drawOnCanvas(img);
        # if self.circleSelected:
        #    cv2.circle(img, (400, 70), 30, (255, 0, 0), cv2.FILLED)
        # else:
        #    cv2.circle(img, (400, 70), 30, (0, 0, 255), cv2.FILLED)

    def drawOnCanvas(self, img):
        # myPos, reelPos = self.findPosition(img)
        for p in self.points:
            cv2.circle(img, (p[0], p[1]), 10, (0, 255, 0), cv2.FILLED)

    def clearCanvas(self):
        self.points = []
    """


if __name__ == "__main__":
    previous_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    cap.set(10, brightness)

    while True:
        #path = detector.select_choice()
        success, img = cap.read()
        img = cv2.flip(img, 1, 1)

        whiteImg = np.ones((img.shape[0], img.shape[1]))
        img = detector.find_hand(img)
        d = detector.compute_distance_between_two_fingers(img, 8, 7)
        pos, reel = detector.find_position(img)

        if(len(pos) != 0):
            index_positions = detector.get_finger_position(img, 8)
            #print(index_positions)
            detector.choice_state(img)
            path = detector.select_choice(index_positions[0], index_positions[1])
            print(path)
            img[0:480, 0:125] = read_img(path)

        # detector.find_position(img, hand_index=12)
        # detector.drawLine(img)
        # detector.selectCircle(whiteImg)
        cv2.putText(img, f'FPS : {int(compute_fps(previous_time))}', (450, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                    2)
        previous_time = time.time()

        cv2.imshow("window", img)
        cv2.imshow("white", whiteImg)
        if quit_program("q"):
            break
