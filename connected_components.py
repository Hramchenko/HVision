import cv2 
import scipy as sp
import matplotlib.pyplot as plt

def component_box(stats, idx):
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]  
    return x, y, w, h

