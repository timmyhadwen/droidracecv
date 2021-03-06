# -*- coding: utf-8 -*-
import cv2 
import numpy as np
import socket
import time 
ADDRESS = "10.1.1.2"
PORT = 3000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ADDRESS, PORT))

def hsvAverage(img, MagicY):
    h = 0.0;
    s = 0.0;
    v = 0.0;
    c = 0.0;
    for i in range(MagicY[1], MagicY[3]):
        for j in range(MagicY[0], MagicY[2]):
            c = c + 1;
            h = h + img[i,j,0]
            s = s + img[i,j,1]
            v = v + img[i,j,2]

    AVY_h = h / c;
    AVY_s = s / c;
    AVY_v = v / c;
    return [AVY_h, AVY_s, AVY_v]

#==============================================================================
# Magical Values 
#==============================================================================
Er = 0.3
gain = 0.8
#==============================================================================
# Camera Setup 
#==============================================================================
CamBlue = cv2.VideoCapture(1)
CamYellow = cv2.VideoCapture(0)
if (CamBlue.isOpened() == False or CamYellow.isOpened() == False):
    CamBlue.release();
    CamYellow.release();
    exit(-1);
CamBlue.set(4,288)
CamBlue.set(3,544)
time.sleep(1)

CamYellow.set(4,288)
CamYellow.set(3,544)

#==============================================================================
#Set Magic Squares 
#==============================================================================
MagicB = [0,220,120,288]#(0,1) top left (2,3) Bottom Right 
#With (X,Y)
MagicY = [425,220,535,287]
# MagicY = MagicB
#CropB = [300:370,0:-1]
#==============================================================================
# Main 
#==============================================================================
while(1):
    sB, frameB = CamBlue.read()
    time.sleep(0.05)
#    _, frameY = CamBlue.read()

    sY, frameY = CamYellow.read()

    if (sY == False or sB == False):
        CamYellow.release()
        CamBlue.release()
        break;


    HSVB = cv2.cvtColor(frameB,cv2.COLOR_BGR2HSV)
    HSVY = cv2.cvtColor(frameY,cv2.COLOR_BGR2HSV)

#==============================================================================
#     Showing Blue for testing 
#==============================================================================
    showblue = frameB
    showyellow = frameY
    cv2.rectangle(showblue,(MagicB[0],MagicB[1]),(MagicB[2],MagicB[3]),(0,255,0),3)
    cv2.rectangle(showyellow,(MagicY[0],MagicY[1]),(MagicY[2],MagicY[3]),(0,255,0),3)
    # cv2.imshow('Line Up Blue',showblue)
    # cv2.imshow('Line Up Not Blue',showyellow)

#==============================================================================
# Average of selection
#==============================================================================
    [AVY_h,AVY_s,AVY_v] = hsvAverage(HSVY, MagicY);
    [AVB_h,AVB_s,AVB_v] = hsvAverage(HSVB, MagicB);

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

    cv2.imshow('Result Blue',resb)
    cv2.imshow('Result Yellow',resy)

#==============================================================================
# Angle magic via maths and shit
#==============================================================================
crop_b = mask_b[100:200,90:-1]
    try:
        top = np.nonzero(crop_b[10,:])[0][0] if np.sum(crop_b[0,:])>0 else 0
#    s2_b = np.nonzero(crop_b[20,:])[0][0] if np.sum(crop_b[20,:])>0 else 544-90
#    s3_b = np.nonzero(crop_b[40,:])[0][0] if np.sum(crop_b[40,:])>0 else 544-90
#    s4_b = np.nonzero(crop_b[60,:])[0][0] if np.sum(crop_b[60,:])>0 else 544-90
#    s5_b = np.nonzero(crop_b[80,:])[0][0] if np.sum(crop_b[80,:])>0 else 544-90
        bot = np.nonzero(crop_b[99,:])[0][0] if np.sum(crop_b[99,:])>0 else 100000
    except:
        bot = 100000
        pass

    cv2.imshow('Praise Matlab',crop_b)
    if top-50>bot:
        DirB = 40
    elif bot == 100000:
        DirB = 55
    else:
        DirB = bot/300*50*gain
        print(top,bot,bot/300*50*gain)
    val = "S{}".format(DirB)
    s.send(val.encode())
 
#==============================================================================
#     Exit Shit 
#==============================================================================
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
CamBlue.release()
CamYellow.release()
cv2.destroyAllWindows()
s.close