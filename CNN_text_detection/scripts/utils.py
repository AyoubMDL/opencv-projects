import cv2
import time


def quit_program(key):
    return cv2.waitKey(1) & 0xFF == ord(key)


def compute_fps(previous_time):
    return 1 / (time.time() - previous_time)
