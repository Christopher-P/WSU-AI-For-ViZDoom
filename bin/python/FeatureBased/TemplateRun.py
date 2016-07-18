#!/usr/bin/python

import threading
from time import sleep

import time

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from vizdoom import *

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

# doom stuff
print "Initializing doom..."
game = DoomGame()
game.load_config("../../../examples/config/learning.cfg")
# Enables freelook in engine
game.add_game_args("+freelook 1")

game.set_window_visible(True)

game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

game.set_mode(Mode.PLAYER)
game.init()
print "Doom initialized."

template = cv2.imread('HEADA1.png', 0)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
threshold = 0.2
plt.ion()

net = buildNetwork(3, 6, game.get_available_buttons_size())

print template.shape[::-1]

image = game.get_game_screen()
finalImage = image

training = True
detected = True
avgX = 0.0
avgY = 0.0

ds = SupervisedDataSet(3, game.get_available_buttons_size())

def aiTrain():
    global detected
    global avgX
    global avgY
    global ds
    while training:
        if np.count_nonzero(game.get_last_action()) > 0:
            ds.addSample((detected, avgX, avgY), game.get_last_action())
            print "Trained!"
            print avgX
            print avgY
        sleep(0.1)


def callDetect():
    while True:
        global image
        global avgX
        global avgY
        clone = image
        #image = game.get_game_screen()
        gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        found = None
        # loop over the scales of the image
        for scale in np.linspace(0.2, 2.4, 10)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
                # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then ipdate
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)


        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # draw a bounding box around the detected result and display the image
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 255), 2)

        global finalImage
        finalImage = clone

        avgX = (startX + endX) / 2
        avgY = (startY + endY) / 2

for i in range (0, 3):
    t = threading.Thread(target=callDetect, args=())
    t.start()

ScreenWidth = game.get_screen_width()
ScreenHeight = game.get_screen_height()
# sleep_time = 0.028
sleep_time = 0.028

net = NetworkReader.readFrom('net.xml')

episodes = 60
avgScore = 0.0
for i in range(episodes):

    print("Episode #" + str(i + 1))

    start = time.time()
    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

        cv2.imshow("final", finalImage)
        cv2.waitKey(1)
        # Gets the state
        s = game.get_state()
        if (time.time() - start) > 0.1:
            image = game.get_game_screen()
            start = time.time()

        action = net.activate([detected, avgX, avgY])
        intAction = []
        print action
        for ind, val in enumerate(action):
            if action[ind] > 0.0001 and action[ind] < 1:
                intAction.append(1)
            elif action[ind] < -0.0001 and action[ind] > -1:
                intAction.append(-1)
            else:
                intAction.append(int(round(action[ind])))
            action[ind] = int(round(action[ind], 0))
            if val < 0:
                action[ind] = 0
            if val > 1:
                action[ind] = 1
            if ind < 16 and ind > 9:
                action[ind] = 0


        # Makes a random action and get remember reward.
        print intAction
        r = game.make_action(intAction, 1)
        a = game.get_last_action()

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