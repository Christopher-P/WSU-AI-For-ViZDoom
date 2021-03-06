#!/usr/bin/python

import itertools as it
import pickle
from random import sample, randint, random, choice
from time import time
from vizdoom import *

import threading

import cv2
import numpy as np

from tqdm import *
from time import sleep
from os import listdir

from matplotlib import pyplot as plt

downsampled_x = 640
downsampled_y = int(2 / 3.0 * downsampled_x)
MIN_MATCH_COUNT = 90
detected = False
location = (-1, -1)
actions = [[True, False, False], [False, True, False], [False, False, True]]
# Removed enemies: 'SPIDA1.png'
enemyName = ['HEADA2.png', 'BOS2H2.png', 'BSPIC5F5.png', 'CPOSB1.png', 'CPOSB5.png', 'CYBRB1.png', 'FATTA1.png',
             'PLAYA1.png', 'POSSC1.png', 'SARGA1.png', 'SKELA1D1.png', 'SKULC1.png', 'SPOSA1.png', 'SSWVF1.png',
             'TROOC1.png', 'VILEB1E1.png']
itemName = ['MEDIA0.png', 'STIMA0.png', 'BON1B0.png', 'BPAKA0.png', 'BROKA0.png', 'BFUGA0.png', 'LAUNA0.png',
            'MGUNA0.png', 'PLASA0.png', 'ROCKA0.png', 'SBOXA0.png', 'SGN2A0.png', 'SHELA0.png']
enemySet = []
itemSet = []
itemDetected = []


# Function for loading images from files
def initTrainSet():
    for ind, val in enumerate(enemyName):
        enemySet.append(cv2.imread(enemyName[ind], 1))

    for ind, val in enumerate(itemName):
        itemSet.append(cv2.imread(itemName[ind], 1))
        t1 = (-1, -1)
        itemDetected.append(t1)  # start with no detected items


# Function for converting images
def convert(img):
    img = img[0].astype(np.float32) / 255.0
    img = cv2.resize(img, (downsampled_x, downsampled_y))
    return img


# Just calling detectimg
def callDetect():
    while True:
        detectImg()


# Function for finding matches between images and game screen
def detectImg():
    img2 = game.get_game_screen()
    kp2, des2 = orb.detectAndCompute(img2, None)
    print "KP2 ", len(kp2)
    for ind, val in enumerate(enemyName):
        img1 = enemySet[ind]
        kp1, des1 = orb.detectAndCompute(img1, None)
        if not des2 is None:
            matches = bf.match(des1, des2)
        else:
            matches = []

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        good = []
        points = []
        points2 = []
        # 90 for no background
        # 10 for regular
        if len(matches) > MIN_MATCH_COUNT:
           # print "Matches found: ", len(matches)
            for i in range(0, MIN_MATCH_COUNT):
                good.append(matches[i])

        if len(matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, depth = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            avgX = 0.0
            for b in dst:
                avgX = avgX + b[0][0]
            avgX = avgX / dst.size * 2

            avgY = 0.0
            for b in dst:
                avgY = avgY + b[0][1]
            avgY = avgY / dst.size * 2

            # avgX = dst[0][0][0]
            # avgY = dst[0][0][1]
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            img2 = cv2.line(img2, (0,0), (int(avgX), int(avgY)),255,3,cv2.LINE_AA)
            # print "Matches Found-----------------------: - %d/%d" % (len(matches),MIN_MATCH_COUNT)

            global detected
            detected = True
            global location
            location = (avgX, avgY)
            #print "Found---------"
            #print val

        else:
            # print "Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT)
            matchesMask = None

            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            #           singlePointColor = None,
            #           matchesMask = matchesMask, # draw only inliers
            #           flags = 2)

            # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        # Display the resulting frame
            cv2.imshow(val,img2)
        # Hold so it has time to draw
            cv2.waitKey(1)
    #print "Finished detect"



    for ind, val in enumerate(itemName):
        img1 = itemSet[ind]
        kp1, des1 = orb.detectAndCompute(img1, None)
        if not des2 is None:
            matches = bf.match(des1, des2)
        else:
            matches = []

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        good = []
        points = []
        points2 = []
        # 90 for no background
        # 10 for regular
        if (len(matches) > MIN_MATCH_COUNT):
            for i in range(0, MIN_MATCH_COUNT):
                good.append(matches[i])

        if len(matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, depth = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            avgX = 0.0
            for b in dst:
                avgX = avgX + b[0][0]
            avgX = avgX / dst.size * 2

            avgY = 0.0
            for b in dst:
                avgY = avgY + b[0][1]
            avgY = avgY / dst.size * 2

            itemDetected[ind] = (avgX, avgY)
            #print "Found---------"
            #print val
        else:
            itemDetected[ind] = (-1, -1)
    #print "Finished items"


# doom stuff
print "Initializing doom..."
game = DoomGame()
game.load_config("../../../examples/config/learning.cfg")
# Enables freelook in engine
game.add_game_args("+freelook 1")

game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(True)

game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

game.set_mode(Mode.SPECTATOR)
game.init()
print "Doom initialized."

initTrainSet()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
orb = cv2.ORB_create()

plt.ion()

t = threading.Thread(target=callDetect, args=())
t.deamon = True
t.start()

episodes = 1000
print("")

ScreenWidth = game.get_screen_width()
ScreenHeight = game.get_screen_height()
action = [False, False, False]
actions = [[True, False, False], [False, True, False], [False, False, True]]
# sleep_time = 0.028
sleep_time = 0.028

avgScore = 0.0
for i in range(episodes):

    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

        if detected:
            x, y = location
            if (x < ((ScreenWidth / 2) - 7)):
                #    print "Moving right"
                action = actions[0]
            elif (x > ((ScreenWidth / 2) + 7)):
                #    print "Moving left"
                action = actions[1]
            else:
                #   print "Shooting"
                action = actions[2]
        # if
        # Gets the state
        s = game.get_state()

        # Makes a random action and get remember reward.
        r = game.make_action(action, 1)
        # r = game.make_action(choice(actions), 1)
        # Prints state's game variables. Printing the image is quite pointless.
        # print("State #" + str(s.number))
        # print("Game variables:", s.game_variables[0])
        # print("Reward:", r)
        # print("=====================")

        if sleep_time > 0:
            sleep(sleep_time)

    # Check how the episode went.
    print("Episode finished.")
    print("total reward:", game.get_total_reward())
    print("************************")
    avgScore = avgScore + game.get_total_reward()

print("Average score is------")
avgScore = avgScore / episodes
print avgScore
game.close()
