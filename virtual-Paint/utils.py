import cv2
import time
import os


def quit_program(key):
    return cv2.waitKey(1) & 0xFF == ord(key)


def compute_fps(previous_time):
    return 1 / (time.time() - previous_time)


def read_img(path):
    return cv2.imread("ressources/"+path)
