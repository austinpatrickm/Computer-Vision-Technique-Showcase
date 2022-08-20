#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:33:01 2021

@author: austin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:51:58 2021

@author: austin
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import sys 


def rescaleFrame(frame, scale=0.3):
    width = (frame.shape[1] * scale)
    height = (frame.shape[0] * scale)
    return cv2.resize(frame, (int(width), int(height)), fx=1, fy=1, interpolation = cv2.INTER_AREA)

font = cv2.FONT_HERSHEY_DUPLEX

def text(inputtext, frametype, xpos, ypos, color=(255,255,255)):
    return cv2.putText(frametype, text= '{}'.format(inputtext), org=(xpos,ypos), fontFace=font, fontScale=2, color=color, thickness=3, lineType=cv2.LINE_AA)
    



cap = cv2.VideoCapture('/Users/austin/Downloads/cloaktest.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(round(cap.get(5)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('/Users/austin/Desktop/CloakoutFINAL4.mp4', fourcc, fps, (576, 324), True)


active_frame = 0


for i in range(5):
    ret, background = cap.read() 
    if ret == False : 
        continue 

background = np.flip(background, axis = 1)



fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while cap.isOpened():
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break 

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = np.flip(frame, axis=1)

      
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    frame_resized = rescaleFrame(frame)
    
    lower_blue1 = np.array([50, 75, 50])
    upper_blue1 = np.array([180, 255, 255])
    lower_blue2 = np.array([105, 75, 50])
    upper_blue2 = np.array([100, 255, 255])
    mask1 = cv2.inRange(bgr, lower_blue1, upper_blue1) 
    mask2 = cv2.inRange(bgr, lower_blue2, upper_blue2)
    mask1 = mask1 + mask2
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations = 1)
    mask2 = cv2.bitwise_not(mask1)
    result1 = cv2.bitwise_and(frame, frame, mask=mask2)
    result2 = cv2.bitwise_and(background, background, mask=mask1)
    
    finalOutput = cv2.addWeighted(result1, 1, result2, 1, 0)
    text('Cloaking', finalOutput, 50, 1000, color=(0, 0, 0))

    
    frame_resized = rescaleFrame(finalOutput)
    
    # mask_resized = rescaleFrame(mask)
    # result_resized = rescaleFrame(result)
     

    cv2.imshow('frame', frame_resized)
    out.write(frame_resized)
    # cv2.imshow('frame1', new_frame)
    # cv2.imshow('mask', mask_resized)
    # cv2.imshow('result', result_resized)
    


    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    
    

cap.release()
out.release()
cv2.destroyAllWindows()