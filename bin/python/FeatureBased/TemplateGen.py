#!/usr/bin/python

import sys
import threading
import time
import csv
from time import sleep

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.xml.networkwriter import NetworkWriter
from skimage.measure import compare_ssim as ssim
from vizdoom import *

# doom stuff
print "Initializing doom..."
game = DoomGame()
game.load_config("../../../examples/config/TemplateGen.cfg")
game.init()
print "Doom initialized."

template = cv2.imread('HEADA1.png', 0)
original = template
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
plt.ion()

net = buildNetwork(3, 6, game.get_available_buttons_size())

image = game.get_game_screen()
finalImage = image

training = True
detected = True
avgX = 0.0
avgY = 0.0

ds = SupervisedDataSet(3, game.get_available_buttons_size())

# Output to CSV ------------------------


with open(__file__, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
        #for line in data:
         #   writer.writerow(line)

data = ["first_name,last_name,city".split(","),

            "Tyrese,Hirthe,Strackeport".split(","),

            "Jules,Dicki,Lake Nickolasville".split(","),

            "Dedric,Medhurst,Stiedemannberg".split(",")

            ]
writer.writerow([10, 10, 0, 10, 10])

def ai_train():
    global detected
    global avgX
    global avgY
    global ds
    while training:
        if np.count_nonzero(game.get_last_action()) > 0:
            ds.addSample((detected, avgX, avgY), game.get_last_action())
            #print "Trained!"
            #print avgX
            #print avgY
        sleep(0.1)

def call_detect():
    global finalImage
    global avgX
    global avgY
    global detected

    h, w = template.shape

    while True:
        #start = time.time()
        clone = image
        gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        found = None
        # loop over the scales of the image
        for scale in np.linspace(2.4, 0.2, 10)[::-1]:
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

                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

                resized = gray[startY:endY, startX:endX]

                resized = cv2.resize(resized, (w, h))

                if ssim(original, resized) > 0.2:
                    break

        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio

        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        resized = gray[startY:endY, startX:endX]

        resized = cv2.resize(resized, (w, h))
        if ssim(original, resized) > 0.2:
            # draw a bounding box around the detected result and display the image
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 255), 2)
            finalImage = clone
            avgX = (startX + endX) / 2
            avgY = (startY + endY) / 2
            detected = True
        else:
            detected = False

        #print '\r',  # print is Ok, and comma is needed.
        #print "Running at: ", 1 / (time.time() - start), " FPS!",
        #sys.stdout.flush()  # flush is needed.

t = threading.Thread(target=call_detect, args=())
t.start()

l = threading.Thread(target=ai_train, args=())
l.start()

action = [False, False, False]
actions = [[True, False, False], [False, True, False], [False, False, True]]
sleep_time = 0.028

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
        # Makes a random action and get remember reward.
        r = game.make_action(action, 1)
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

print "Game Finished, Starting Training..."

trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence()
NetworkWriter.writeToFile(net, 'net.xml')

print "-----------------------------"
print "DONE"
