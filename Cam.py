import cv2 
import numpy as np
import socket

ADDRESS = "10.1.1.2"
PORT = 3000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ADDRESS, PORT))

#cap.release
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('C:/Users/Rudge-Top/Desktop/lap2-1321.mov')

#while(1):
while(cap.isOpened()):


    # Take each frame
    _, frame = cap.read()
    frame = frame[400:-1,-400:-1]
    
    #COnvert Frame to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
#    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)

    #Apply Gaussian Blur to the Frame 
    Blur = cv2.GaussianBlur(hsv,(5,5),0) #This is all that is required for a Guassian Blur 
#==============================================================================
#     HSV
#==============================================================================
    #Create a Mask to Find the Colour Blue 
    lower_blue = np.array([34,129,200])
    upper_blue = np.array([109,248,250])
    lower_yellow =np.array([20,60,50])
    upper_yellow = np.array([55,200,255])
#==============================================================================
# LAB 
#==============================================================================
#    lower_blue = np.array([31*(255/100),-10+128,-45+128])
#    upper_blue = np.array([42*(255/100),6+128,-20+128])
#    lower_blue = np.array([31,-10,-45])
#    upper_blue = np.array([42,6,-27])

#    lower_yellow = np.array([37*(255/100),-10+128,50+128])
#    upper_yellow = np.array([67*(255/100),2+128,61+128])
#==============================================================================
#     
#==============================================================================
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5,5),np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
#    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

#    mask = cv2.erode(mask,kernel,iterations = 1)
#    mask = cv2.GaussianBlur(mask,(5,5),0) #This is all that is required for a Guassian Blur 
#    mask = cv2.medianBlur(mask,5)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res2 = cv2.bitwise_and(frame,frame, mask= mask2)
    try:
        idx = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
        idx_y = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
#        print(cv2.contourArea(idx))
        if cv2.contourArea(idx_y)<300:
            [vx_y,vy_y] = [1,1]
        else :
            [vx_y,vy_y,x_y,y_y] = cv2.fitLine(idx_y, cv2.DIST_L2,0,0.05,0.05)
            print(x_y,y_y)
        if cv2.contourArea(idx)<300:
            [vx,vy] = [1,1]
        else:    
            [vx,vy,x,y] = cv2.fitLine(idx, cv2.DIST_L2,0,0.1,0.1)
            print(x,y)
    except:
        [vx,vy,vx_y,vy_y] = [1,0,1,0]
        pass
#    print("Angle Blue:\t",np.rad2decg(np.arctan(vy/vx)))
    showme = cv2.bitwise_or(res,res2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    B_Angle = np.rad2deg(np.arctan(vy/vx))
    Y_Angle = np.rad2deg(np.arctan(vy_y/vx_y))
#  
    if B_Angle == 45:
        DirB = 10.00
    elif B_Angle == 0:
        DirB = 45
    else:
        DirB = 90+(B_Angle*50/45)
#    cv2.putText(showme, "%.2f"%B_Angle ,(100,100), font, 1,(0,0,255),2,cv2.LINE_AA)
#    cv2.putText(showme, "%.2f"%Y_Angle,(400,100), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(showme,"%.2f"%DirB,(250,100), font, 1,(255,255,255),2,cv2.LINE_AA)

<<<<<<< HEAD
    cv2.imshow('Hardcore memery',res2)
=======
    val = "S{}".format(DirB)
    s.send(val.encode())

    cv2.imshow('Hardcore memery',showme)
>>>>>>> 33e915d0a57d4638bd1f0f40be37ad084b424335
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
<<<<<<< HEAD
del k
=======
s.close()
#del k
>>>>>>> 33e915d0a57d4638bd1f0f40be37ad084b424335
