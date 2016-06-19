#!/usr/bin/python

import itertools as it
import pickle
from random import sample, randint, random
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
MIN_MATCH_COUNT = 10

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
        print "Worked..."
        img4 = game.get_game_screen()
#        img4 = np.rollaxis(img4, 0, 3)
        kp2, des2 = orb.detectAndCompute(img4,None)
        matches = bf.match(des1,des2)

	#Sort them in the order of their distance.
         #matches = sorted(matches, key = lambda x:x.distance)

	#Draw first 10 matches
        img3 = np.zeros((10,10,3), np.uint8)
        img3 = cv2.drawMatches(img1,kp1,img4,kp2,matches[:10], img3)

    #cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3)

 #   # Capture frame-by-frame
        #ret, frame = cap.read()
 #   # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
        cv2.imshow('frame',img3)
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
game.set_mode(Mode.SPECTATOR)
game.init()
print "Doom initialized."

#sleep(10)

#FROM: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#AUTHOR: 

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

episodes = 2
print("")

for i in range(episodes):
    
    #kp2, des2 = orb.detectAndCompute(img2,None)
    #matches = bf.match(des1,des2)

	#Sort them in the order of their distance.
    #matches = sorted(matches, key = lambda x:x.distance)

	#Draw first 10 matches
    #img3 = np.zeros((10,10,3), np.uint8)
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3)

    #cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3)
    #plt.imshow(img3)

    print("Episode #" +str(i+1))
    game.new_episode()
    while not game.is_episode_finished():

        s = game.get_state()
        img = s.image_buffer
        misc = s.game_variables

        game.advance_action()
        a = game.get_last_action()
        r = game.get_last_reward()

        #print("state #"+str(s.number))
        #print("game variables: ", misc)
        #print("action:", a)
        #print("reward:",r)
        #print("=====================")

    print("episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

game.close()