#!/usr/bin/python

import itertools as it
import pickle
from random import sample, randint, random, choice
from time import time
from vizdoom import *

import threading

import cv2
import numpy as np
import theano
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, MaxPool2DLayer, get_output, get_all_params, \
    get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from theano import tensor
from tqdm import *
from time import sleep

from matplotlib import pyplot as plt

downsampled_x = 640
downsampled_y = int(2/3.0*downsampled_x)
MIN_MATCH_COUNT = 90
detected = False
location = (-1,-1)



actions = [[True,False,False],[False,True,False],[False,False,True]]

# Function for converting images
def convert(img):
    img = img[0].astype(np.float32) / 255.0
    img = cv2.resize(img, (downsampled_x, downsampled_y))
    return img
	
#def imgThread( p ):
def imgThread(  ):
    count = 0
    cap = cv2.VideoCapture(0)
    while(True):
        #print "Worked..."
        img2 = game.get_game_screen()
        kp2, des2 = orb.detectAndCompute(img2,None)
        if not des2 is None:
            matches = bf.match(des1,des2)
        else:
            matches = []
		
		# Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        ##img_rgb = cv2.imread('mario.png')
        #img_rgb = img2
        #img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        ##template = cv2.imread('mario_coin.png',0)
        #template = img1
        #w, h, dim = template.shape[::-1]
		
        #res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
        #threshold = 0.2
        #loc = np.where( res >= threshold)
        #for pt in zip(*loc[::-1]):
        #    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
			
        #cv2.imwrite('res.png',img_rgb)
    
        #cv2.imwrite('res.png',img2)
        #print "Matches"
        #print 
		#for i in matches:
            
	#Sort them in the order of their distance.
         #matches = sorted(matches, key = lambda x:x.distance)

	#Draw first 10 matches
#------------------------------------------------Bounding BOx
        #print "Matches[0]"
        #print matches[0].shape
        #good = matches
        good = []
        points = []
        points2 = []
        #for m in matches:
        #    if m.distance < 0.7*50:
        #        good.append(m)
        if (len(matches) > 20):
            for i in range (0,20):
                good.append(matches[i])
		
		
        if len(matches)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            #print img1.shape
            h,w, depth = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            #pts = np.float32([ [0,0],[0,60-1],[50-1,60-1],[50-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            avgX  = 0.0
            for b in dst_pts:
                avgX = avgX + b[0][0]
            avgX = avgX / dst_pts.size * 2
			
            avgY  = 0.0
            for b in dst_pts:
                avgY = avgY + b[0][1]
            avgY = avgY / dst_pts.size * 2

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            img2 = cv2.line(img2, (0,0), (int(avgX), int(avgY)),255,3,cv2.LINE_AA)
            #print "Matches Found-----------------------: - %d/%d" % (len(matches),MIN_MATCH_COUNT)
			
            global detected
            detected = True
            global location
            location = (avgX,avgY)
			
            #for n in good:
            points.append(good[0].queryIdx)
            points2.append(good[0].trainIdx)
            print(points[0])
            print(points2[0])
            #print(matches[0].distance)

        else:
            print "Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT)
            matchesMask = None


	
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        #img3 = np.zeros((10,10,3), np.uint8)
        #img3 = cv2.drawMatches(img1,kp1,img4,kp2,matches[:10], img3)
#--------------------------------------
    #cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3)

 #   # Capture frame-by-frame
        #ret, frame = cap.read()
 #   # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
        cv2.imshow('frame',img3)
        #cv2.imshow('frame',img_rgb)
		#cv2.imshow('frame', game.get_state().image_buffer)
        cv2.waitKey(1)

def imgThread2():
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('box.png',0)          # queryImage
    img2 = game.get_game_screen()
    img2 = np.rollaxis(img2, 0, 3)

# Initiate SIFT detector
    sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
		
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
		
	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()
#    while True:
#        count += 1
#        print "Grabbing Image..."
#        img4 = game.get_game_screen()
#        img4 = np.rollaxis(img4, 0, 3)
#        gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
#        plt.imshow(img4)
 ##       cv2.waitKey(0);
#        #p.set_data(img4)
#        #plt.pause(0.5)
#        #plt.imshow(img4)
#        #plt.pause(1.0)
#        print "Grabbed Image!"
#        sleep(0.001)
		
		
#fig = plt.gcf()
#fig.show()
#fig.canvas.draw()

#while True:
#    plt.plot([1], [2])
#	
#    plt.xlim([0, 100])
#    plt.ylim([0, 100])
	
#    fig.canvas.draw()
		
#doom stuff
print "Initializing doom..."
game = DoomGame()
game.load_config("../../examples/config/learning.cfg")
# Enables freelook in engine
game.add_game_args("+freelook 1")

game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(True)

game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

game.init()
print "Doom initialized."

#sleep(10)

#FROM: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#AUTHOR: 
#img1 = cv2.imread('box.png',0)          # queryImage
#--------
img1 = cv2.imread('HEADB1.png',1)          # queryImage
#img1 = (255-img1)
#img2 = cv2.imread('box_in_scene.png',-1)
img2 = game.get_game_screen()
#img2 = np.rollaxis(img2, 0, 3) 
#img2 = convert(game.get_state().image_buffer) 
print "---------"
print img2.shape
print "---------"
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#img2 = cv2.imread(convert(game.get_state().image_buffer),0) # trainImage

#plt.imshow(img1),plt.show()
#plt.imshow(img2),plt.show()

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

#good = []
#for m in matches:
#    if m.distance < 0.7:
#        good.append(m)
		
#if len(good)>MIN_MATCH_COUNT:
#    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#    matchesMask = mask.ravel().tolist()

#    h,w = img1.shape
#    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#    dst = cv2.perspectiveTransform(pts,M)

#    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

#else:
#    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#    matchesMask = None
	
#draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                   singlePointColor = None,
#                   matchesMask = matchesMask, # draw only inliers
#                   flags = 2)

#img3 = np.zeros((10,10,3), np.uint8)
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,img3,**draw_params)

#plt.imshow(img3, 'gray'),plt.show()

# Draw first 10 matches
img3 = np.zeros((10,10,3), np.uint8)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3)

#img3 = img1
#cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3)


plt.ion()
p = plt.imshow(img3)
fig = plt.gcf()
plt.title("Boring slide show")
plt.pause(0.5)

#t = threading.Thread(target=imgThread, args=(p,))
t = threading.Thread(target=imgThread, args=())
t.deamon = True
t.start()

episodes = 20
print("")

ScreenWidth = game.get_screen_width()
ScreenHeight = game.get_screen_height()
action = [False, False, False]
actions = [[True,False,False],[False,True,False],[False,False,True]]
sleep_time = 0.028

for i in range(episodes):
    
    print("Episode #" + str(i+1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

        if detected:
            x,y = location
            print "X is "
            print x
            if(x < ((ScreenWidth / 2) - 7)):
            #    print "Moving right"
                action = actions[0]
            elif(x > ((ScreenWidth / 2) + 7)):
            #    print "Moving left"
                action = actions[1]
            else:
             #   print "Shooting"
                action = actions[2]
        print action
        print detected
        # Gets the state
        s = game.get_state()

        # Makes a random action and get remember reward.
        r = game.make_action(action, 1)
        #r = game.make_action(choice(actions), 1)
        # Prints state's game variables. Printing the image is quite pointless.
        print("State #" + str(s.number))
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

        if sleep_time>0:
            sleep(sleep_time)

    # Check how the episode went.
    print("Episode finished.")
    print("total reward:", game.get_total_reward())
    print("************************")

game.close()