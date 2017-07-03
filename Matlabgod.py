# -*- coding: utf-8 -*-
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import time 
#==============================================================================
# Magical Values 
#==============================================================================
Er = 0.4
gain = 0.7


#==============================================================================
# Camera Setup 
#==============================================================================
CamBlue = cv2.VideoCapture(0)
CamBlue.set(4,288)
CamBlue.set(3,544)
CamYellow = cv2.VideoCapture(1)
CamYellow.set(4,288)
CamYellow.set(4,544)

#==============================================================================
#Set Magic Squares 
#==============================================================================
MagicB = [0,220,120,288]#(0,1) top left (2,3) Bottom Right 
#With (X,Y)
MagicY = [535-110,240,535,288]
#CropB = [300:370,0:-1]
#==============================================================================
# Main 
#==============================================================================
while(1):
    _, frameB = CamBlue.read()
    print()
    time.sleep(0.05)
    _, frameY = CamBlue.read()

#    _, frameY = CamYellow.read()
    HSVB = cv2.cvtColor(frameB,cv2.COLOR_BGR2HSV)
    HSVY = cv2.cvtColor(frameY,cv2.COLOR_BGR2HSV)

#==============================================================================
#     Showing Blue for testing 
#==============================================================================
    showblue = frameB
    showyellow = frameY
    cv2.rectangle(showblue,(MagicB[0],MagicB[1]),(MagicB[2],MagicB[3]),(0,255,0),3)
    cv2.rectangle(showyellow,(MagicY[0],MagicY[1]),(MagicY[2],MagicY[3]),(0,255,0),3)
    cv2.imshow('Line Up Blue',showblue)
    cv2.imshow('Line Up Not Blue',showyellow)

#==============================================================================
# Average of selection
#==============================================================================
    AVB_h = np.mean(HSVB[MagicB[1]:MagicB[3],MagicB[0]:MagicB[2]][0])
    AVB_s = np.mean(HSVB[MagicB[1]:MagicB[3],MagicB[0]:MagicB[2]][1])
    AVB_v = np.mean(HSVB[MagicB[1]:MagicB[3],MagicB[0]:MagicB[2]][2])
#    print([AVB_h,AVB_s,AVB_v])
    AVY_h = np.mean(HSVY[MagicY[1]:MagicY[3],MagicY[0]:MagicY[2]][0])
    AVY_s = np.mean(HSVY[MagicY[1]:MagicY[3],MagicY[0]:MagicY[2]][1])
    AVY_v = np.mean(HSVY[MagicY[1]:MagicY[3],MagicY[0]:MagicY[2]][2])
    
    LowB =np.array([AVB_h-Er*AVB_h,AVB_s-Er*AVB_s,AVB_v-Er*AVB_v])
    UpB = np.array([AVB_h+Er*AVB_h,AVB_s+Er*AVB_s,AVB_v+Er*AVB_v])
    mask_b = cv2.inRange(HSVB,LowB,UpB)
    
    LowY =np.array([AVY_h-Er*AVY_h,AVY_s-Er*AVY_s,AVY_v-Er*AVY_v])
    UpY = np.array([AVY_h+Er*AVY_h,AVY_s+Er*AVY_s,AVY_v+Er*AVY_v])
    mask_y = cv2.inRange(HSVY,LowY,UpY)
    
#==============================================================================
#     Close
#==============================================================================
    kernel = np.ones((5,5),np.uint8)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kernel)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel)

#==============================================================================
#     Show
#==============================================================================
    resb = cv2.bitwise_and(frameB,frameB, mask= mask_b)
    resy = cv2.bitwise_and(frameY,frameY, mask= mask_y)

#    cv2.imshow('Praise Matlab',resb)
#    cv2.imshow('Praise Matlab',resy)

#==============================================================================
# Angle magic via maths and shit
#==============================================================================
    crop_b = mask_b[100:200,90:-1]    
    bot = np.nonzero(crop_b[0,:])[0][0] if np.sum(crop_b[0,:])>0 else 100000
#    s2_b = np.nonzero(crop_b[20,:])[0][0] if np.sum(crop_b[20,:])>0 else 544-90
#    s3_b = np.nonzero(crop_b[40,:])[0][0] if np.sum(crop_b[40,:])>0 else 544-90
#    s4_b = np.nonzero(crop_b[60,:])[0][0] if np.sum(crop_b[60,:])>0 else 544-90
#    s5_b = np.nonzero(crop_b[80,:])[0][0] if np.sum(crop_b[80,:])>0 else 544-90
    top = np.nonzero(crop_b[99,:])[0][0] if np.sum(crop_b[99,:])>0 else 100000
#    yb = np.array([99,80,60,40,20,0])
#    xb = np.array([s1_b,s2_b,s3_b,s4_b,s5_b,s6_b])
#    lineb = np.polyfit(xb, yb, 1)
#    angle_b = np.rad2deg(np.arctan(lineb[0])) 
#    plt.plot(xb, yb, 'ro')
#    plt.axis([0, 6, 0, 20])
#    plt.show()
#    print(angle_b)
    cv2.imshow('Praise Matlab',crop_b)
#    print(s1_b,s6_b)
    if top>bot:
        print(35)
    elif bot == 100000:
        print(50)
    else:
        print(bot/240*50*gain)
        
#==============================================================================
#     Exit Shit 
#==============================================================================
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
CamBlue.release()
CamYellow.release()
cv2.destroyAllWindows()
del k

