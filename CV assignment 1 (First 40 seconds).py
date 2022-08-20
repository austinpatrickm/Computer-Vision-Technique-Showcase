#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:33:56 2021

@author: austin
"""





import numpy as np
from matplotlib import pyplot as plt
import cv2
import textwrap
from random import randint

active_frame = 0


cap = cv2.VideoCapture('/Users/austin/Desktop/Assignment1pre.mp4')

capSize = (576, 324)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')



fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
font = cv2.FONT_HERSHEY_DUPLEX   #cv2.FONT_ITALIC


out = cv2.VideoWriter('/Users/austin/Desktop/ComputerVisionOUTPUTFINAL4.mp4', fourcc, fps, (576, 324), True)




def rescaleFrame(frame, scale=0.3): #was .3
    width = (frame.shape[1] * scale)
    height = (frame.shape[0] * scale)
    return cv2.resize(frame, (int(width), int(height)), fx=1, fy=1, interpolation = cv2.INTER_AREA)

def text(inputtext, frametype, xpos, ypos, color=(255, 255, 255), fontScale=1.5):
    return cv2.putText(frametype, text= '{}'.format(inputtext), org=(xpos,ypos), fontFace=font, fontScale=fontScale, color=color, thickness=3, lineType=cv2.LINE_AA)
    

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hsv_lower_blue = np.array([50, 75, 0])
    hsv_upper_blue = np.array([180, 255, 255])
    rgb_lower_blue = np.array([0,0,65]) #0,0,65
    rgb_upper_blue = np.array([75,255,255]) #75, 255, 255
    
    mask1 = cv2.inRange(rgb, rgb_lower_blue, rgb_upper_blue)
    mask1_resized = cv2.resize(mask1, (frame_width, frame_height), cv2.INTER_NEAREST)
    mask1_resized = cv2.cvtColor(mask1_resized, cv2.COLOR_GRAY2RGB)
    mask2 = cv2.inRange(hsv, hsv_lower_blue, hsv_upper_blue)
    mask2_resized = cv2.resize(mask2, (frame_width, frame_height), cv2.INTER_NEAREST)
    mask2_resized = cv2.cvtColor(mask2_resized, cv2.COLOR_GRAY2RGB)
    
    result1 = cv2.bitwise_and(frame, frame, mask=mask1)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)
    
    template = cv2.imread('/Users/austin/Downloads/IMG_3360.jpg', 0) # was Desktop/Orangetemplate3.png
    w, h = template.shape[::-1]
    

    res = cv2.matchTemplate(frame_gray,template,cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(frame_gray,template,cv2.TM_SQDIFF) #
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc 
    bottom_right = (top_left[0] + w, top_left[1] + h)

 
    
    
    if ret:
     
        
        # Gray
        if (fps < active_frame < fps*5):
            if (fps < active_frame < fps*2):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if (fps*2 < active_frame < fps*3):
                pass
            if (fps*3 < active_frame < fps*4):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if (fps*4 < active_frame < fps*5):
                pass
            text('BGR to gray and back', frame, 70, 210)
                
        
        #Filters
        kernel = np.ones((5,5), np.float32)/25
        kernel2 = np.ones((7,7), np.float32)/25
        if ((fps*5)-1 < active_frame < fps*13):
            if ((fps*5)-1 < active_frame < fps*7):
                frame = cv2.GaussianBlur(frame, (11, 11), 0)
                text('Gaussian Blur', frame, 70, 210)
                text('(Kernel 11x11)', frame, 70, 290)
        
            if ((fps*7)-1 < active_frame < fps*9):
                frame = cv2.GaussianBlur(frame, (31, 31), 0)
                text('Gaussian Blur', frame, 70, 210)
                text('(Kernel 31x31)', frame, 70, 290)

            
            if ((fps*9)-1 < active_frame < fps*11):
                frame = cv2.bilateralFilter(frame,11,20,20)
                text('Bilateral Filter', frame, 70, 210)
                text('(Kernel 13x13)', frame, 70, 290)
                
            
            if ((fps*11)-1 < active_frame < fps*13):
                # change d to 30 for 30X30
                frame = cv2.bilateralFilter(frame,29,100,500)
                text('Bilateral Filter', frame, 70, 210)
                text('(Kernel 29x29)', frame, 70, 290)
                
          
        # Grab
        if ((fps*13)-1 < active_frame < fps*21):
            if ((fps*13)-1 < active_frame < fps*14):
                frame = mask1_resized
                text('RBG grab: binary', mask1_resized, 70, 210)
            
            if ((fps*14)-1 < active_frame < fps*15):
                frame = result1
                text('RBG grab', result1, 70, 210)
            
            if ((fps*15)-1 < active_frame < fps*16):
                mask3 = cv2.bitwise_not(mask2_resized)
                frame = mask3
                text('HSV grab: binary', frame, 70, 210, color=(0,0,0))
                text('(flipped)', frame, 70, 290, color=(0,0,0))
                
            if ((fps*16)-1 < active_frame < fps*17):
                frame = result2
                text('HSV grab', result2, 70, 210)
            
            if ((fps*17)-1 < active_frame < fps*18):
                text('Morphological op:', result1, 70, 210)
                text('Dilation', result1, 70, 290)
                frame = cv2.dilate(result1, kernel, dst=None, anchor=None, iterations=1)
            
            if ((fps*18)-1 < active_frame < fps*19):
                frame = cv2.dilate(result2, kernel, dst=None, anchor=None, iterations=1)
                difference = cv2.subtract(frame, result2)
                Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
                difference[mask != 255] = [0, 255, 255]
                frame = difference
                text('Morphological op:', frame, 70, 210, color=(255,255,255), fontScale=1.3)
                text('Dilation', frame, 70, 270, color=(255,255,255), fontScale=1.3)
                text('(Improvement isolated)', frame, 70, 330, color=(255,255,255), fontScale=1.3)
        
            if ((fps*19)-1 < active_frame < fps*20):
                text('Morphological op:', result2, 70, 210)
                text('Opening', result2, 70, 290)
                frame = cv2.morphologyEx(result2, cv2.MORPH_OPEN, kernel2)
                
            if ((fps*20)-1 < active_frame < fps*21):
                frame = cv2.morphologyEx(result2, cv2.MORPH_OPEN, kernel2)                
                difference = cv2.subtract(result2, frame)
                Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
                difference[mask != 255] = [0, 255, 255]
                frame = difference
                text('Morphological op:', frame, 70, 210, color=(255,255,255), fontScale=1.3)
                text('Opening', frame, 70, 270, color=(255,255,255), fontScale=1.3)
                text('(Improvement isolated)', frame, 70, 330, color=(255,255,255), fontScale=1.3)
         
            
        # Sobel
        if ((fps*21)-1 < active_frame < fps*(25)+1):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ((fps*21)-1 < active_frame < fps*22):
                sobel_x_filtered_frame = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3, scale=1)
                sobel_x_filtered_frame = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3, scale=1)
                sobel_y_filtered_frame = cv2.convertScaleAbs(sobel_x_filtered_frame)
                sobel_y_filtered_frame = cv2.convertScaleAbs(sobel_y_filtered_frame)
                sobel_y_filtered_frame = cv2.cvtColor(sobel_y_filtered_frame, cv2.COLOR_GRAY2RGB)
                text('Sobel edge detection', sobel_y_filtered_frame, 70, 210, color=(0,0,255))
                text('(Kernel 3X3, Scale=1)', sobel_y_filtered_frame, 70, 290, color=(0,0,255))
                frame = sobel_y_filtered_frame
            
            if ((fps*22)-1 < active_frame < fps*23):
                sobel_x_filtered_frame = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=5, scale=1)
                sobel_x_filtered_frame = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=5, scale=1)
                sobel_y_filtered_frame = cv2.convertScaleAbs(sobel_x_filtered_frame)
                sobel_y_filtered_frame = cv2.convertScaleAbs(sobel_y_filtered_frame)
                sobel_y_filtered_frame = cv2.cvtColor(sobel_y_filtered_frame, cv2.COLOR_GRAY2RGB)
                text('Sobel edge detection', sobel_y_filtered_frame, 70, 210, color=(0,0,255))
                text('(Kernel 5X5, Scale=1)', sobel_y_filtered_frame, 70, 290, color=(0,0,255))
                frame = sobel_y_filtered_frame
            
            if ((fps*23)-1 < active_frame < fps*24):
                sobel_x_filtered_frame = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3, scale=3)
                sobel_x_filtered_frame = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3, scale=3)
                sobel_y_filtered_frame = cv2.convertScaleAbs(sobel_x_filtered_frame)
                sobel_y_filtered_frame = cv2.convertScaleAbs(sobel_y_filtered_frame)
                sobel_y_filtered_frame = cv2.cvtColor(sobel_y_filtered_frame, cv2.COLOR_GRAY2RGB)
                text('Sobel edge detection', sobel_y_filtered_frame, 70, 210, color=(0, 0, 255))
                text('(Kernel 3X3, Scale=3)', sobel_y_filtered_frame, 70, 290, color=(0, 0, 255))
                frame = sobel_y_filtered_frame
                
            if ((fps*24)-1 < active_frame < (fps*25)+1):
                sobel_x_filtered_frame = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3, scale=6)
                sobel_x_filtered_frame = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3, scale=6)
                sobel_y_filtered_frame = cv2.convertScaleAbs(sobel_x_filtered_frame)
                sobel_y_filtered_frame = cv2.convertScaleAbs(sobel_y_filtered_frame)
                sobel_y_filtered_frame = cv2.cvtColor(sobel_y_filtered_frame, cv2.COLOR_GRAY2RGB)
                text('Sobel edge detection', sobel_y_filtered_frame, 70, 210, color=(0, 0, 255))
                text('(Kernel 3X3, Scale=6)', sobel_y_filtered_frame, 70, 290, color=(0, 0, 255))
                frame = sobel_y_filtered_frame
                
            
            
        # Hough
        if ((fps*25) < active_frame < (fps*35)+1):
            frame_gray = cv2.medianBlur(frame_gray, 5)
            rows = frame_gray.shape[0]
            if ((fps*25) < active_frame < (fps*27)):
                circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, rows / 30,
                                param1=140, param2=30, #was 31
                                minRadius=3, maxRadius=110)
                circles = np.uint16(np.around(circles))
                text('Hough transform', frame, 70, 210, color=(255, 255, 255), fontScale=1.3)
                text('(param1=140, param2=30)', frame, 70, 290, color=(255, 255, 255), fontScale=1.3)
                text('minR=3, maxR=110)', frame, 70, 370, color=(255, 255, 255), fontScale=1.3)
            if ((fps*27)-1 < active_frame < (fps*30)):
                circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, rows / 30,
                                param1=200, param2=45,
                                minRadius=3, maxRadius=190)
                circles = np.uint16(np.around(circles))
                text('Hough transform', frame, 1200, 950, color=(255, 255, 255), fontScale=1)
                text('(param1=250, param2=45)', frame, 1200, 1000, color=(255, 255, 255), fontScale=1)
                text('minR=3, maxR=190)', frame, 1200, 1050, color=(255, 255, 255), fontScale=1)
            if ((fps*30)-1 < active_frame < (fps*33)):
                circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, rows / 30,
                                param1=200, param2=10,
                                minRadius=0, maxRadius=200) #was 3 and 190 for max and min
                circles = np.uint16(np.around(circles))
                text('Hough transform', frame, 1200, 950, color=(255,255,255), fontScale=1)
                text('(param1=200, param2=10)', frame, 1200, 1000, color=(255,255,255), fontScale=1)
                text('minR=0, maxR=200)', frame, 1200, 1050, color=(255, 255, 255), fontScale=1)
            if ((fps*33)-1 < active_frame < (fps*35)):
                circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, rows / 30,
                                param1=100, param2=20,
                                minRadius=10, maxRadius=150)
                circles = np.uint16(np.around(circles))
                text('Hough transform', frame, 1200, 950, color=(255,255,255), fontScale=1)
                text('(param1=100, param2=20)', frame, 1200, 1000, color=(255,255,255), fontScale=1)
                text('minR=10, maxR=150)', frame, 1200, 1050, color=(255,255,255), fontScale=1)
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                # cv2.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(frame, center, radius, (255, 0, 255), 3)
                
                
                
        
        # Template match
        if ((fps*35)+1 < active_frame < (fps*37)):
            text('Template Match', frame, 50, 1000)
            cv2.rectangle(frame,top_left, bottom_right, 255, 2)
                
        
        if ((fps*37)-1 < active_frame < (fps*40)):
            new_frame = cv2.normalize(res2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            new_frame = cv2.cvtColor(cv2.normalize(new_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_GRAY2BGR)
            new_frame = cv2.resize(new_frame, (frame_width, frame_height), cv2.INTER_NEAREST)
            # new_frame = cv2.copyMakeBorder(new_frame, 30, 30, 10, 10, cv2.BORDER_CONSTANT)
            height1, width1 = new_frame.shape[:2]
            # print(height1, width1)
            text('Likelihood Map', new_frame, 50, 1000, color=(0, 0, 255))
            frame = new_frame
            
     
        
        
                    
                    
            

            
    
    
    frame = rescaleFrame(frame)
    res_resized = rescaleFrame(res)
    
    height, width = frame.shape[:2]
    print(height, width)
    
    
    
    # height6, width6 = mask1_resized.shape[:2]
    # print(height6, width6)
    
    
    


        
    cv2.imshow('frame', frame)
    out.write(frame)
    active_frame += 1
    if cv2.waitKey(100) & 0xFF == ord('q'):
            break 

        
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

print(fps)
cap.release()
out.release()
cv2.destroyAllWindows()

