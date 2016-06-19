import datetime
import numpy as np
from PIL import ImageGrab
import cv2
import win32gui

#dimension of image and video
dim=(680,400) 
#video format DIVX
fourcc = cv2.cv.CV_FOURCC(*'DIVX') 
fps=20
#video filename 
now = datetime.datetime.now()
stamp=str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+" (" +str(now.day)+"."+str(now.month)+"."+str(now.year)+")"
filename="video_"+stamp+".avi"
#opencv video recorder 
out = cv2.VideoWriter(filename,fourcc, fps, (680,400))
while(True):
        #take screenshot
        printscreen_pil =  ImageGrab.grab()
        printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype=np.uint8).reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
        #get mouse cursor position and add red circle  
        position = win32gui.GetCursorPos()
        cv2.circle(printscreen_numpy,position, 15, (0,0,255),  -1)
        # resizing of captured image
        printscreen_numpy = cv2.resize(printscreen_numpy, dim, interpolation = cv2.INTER_AREA)
        # create text for time stamp
        now = datetime.datetime.now()
        stamp="---"+str(now.hour)+":"+str(now.minute)+":"+str(now.second)+" ("+str(now.microsecond)+")   "+str(now.day)+"/"+str(now.month)+"/"+str(now.year)+"---"
        cv2.putText(printscreen_numpy,stamp,(200,390), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
        #write image to video file
        out.write(printscreen_numpy)
        # show image
        cv2.imshow('frame',printscreen_numpy)
        #press "q" for exit and save 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# save video
out.release()
# close window
cv2.destroyAllWindows()