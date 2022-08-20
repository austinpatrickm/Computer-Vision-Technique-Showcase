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
from random import randint

def rescaleFrame(frame, scale=0.3):
    width = (frame.shape[1] * scale)
    height = (frame.shape[0] * scale)
    return cv2.resize(frame, (int(width), int(height)), fx=1, fy=1, interpolation = cv2.INTER_AREA)

font = cv2.FONT_HERSHEY_DUPLEX

def text(inputtext, frametype, xpos, ypos, color=(255,255,255)):
    return cv2.putText(frametype, text= '{}'.format(inputtext), org=(xpos,ypos), fontFace=font, fontScale=2, color=color, thickness=3, lineType=cv2.LINE_AA)
    



cap = cv2.VideoCapture('/Users/austin/Desktop/Freestyle.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(round(cap.get(5)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('/Users/austin/Desktop/Freestyleout2.mp4', fourcc, fps, (576, 324), True)


active_frame = 0






fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while cap.isOpened():
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break 

    ret, frame = cap.read()
    
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    text('Tracking a moving object', frame, 70, 100, color=(0,0,0))
    
    
    lower_green1 = np.array([32, 30, 0])
    upper_green1 = np.array([75, 255, 255])
    mask = cv2.inRange(bgr, lower_green1, upper_green1) 
 
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    
    if (0 < active_frame < fps*5):
        gray = cv2.medianBlur(gray, 41)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, .0001, rows / 30,
                                   param1=70.5, param2=37,
                                   minRadius=48, maxRadius=190) #d = .001, param1 = 73.5, param2=41.5, minRadius=48, maxRadius=190
        
        # Param1 = sensitivity: how strong the edges need to be
        # Param2 = how many edge points need to be found for a circle to be declared
        
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                cv2.circle(frame, center, radius+25, (255, 0, 255), 17)
               
    
    if ((fps*5)-1 < active_frame < (fps*7)-15):
        gray = cv2.medianBlur(gray, 27)#or 21
  
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, .0001, rows / 30,
                                   param1=69.5, param2=36.5,
                                   minRadius=48, maxRadius=200) #d = .001, param1 = 73.5, param2=41.5, minRadius=48, maxRadius=190
        
        # Param1 = sensitivity: how strong the edges need to be
        # Param2 = how many edge points need to be found for a circle to be declared
        
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                cv2.rectangle(frame, (i[0]-i[2]-10, i[1]-i[2]-10), (i[0]+i[2]+10, i[1]+i[2]+10), (255, 255, 255), 20)
               
                
        
        

   
    frame[mask > 0] = (randint(0, 255), randint(0, 255), randint(0, 255))
    frame_resized = rescaleFrame(frame)
    height, width = frame_resized.shape[:2]
    out.write(frame_resized)
    cv2.imshow("detected circles", frame_resized)
    active_frame += 1
    
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    
    

cap.release()
out.release()
cv2.destroyAllWindows()