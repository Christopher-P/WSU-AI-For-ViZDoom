#!/usr/bin/python

import threading
from time import sleep

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from vizdoom import *

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

game.set_mode(Mode.SPECTATOR)
game.init()
print "Doom initialized."

template = cv2.imread('HEADA1.png', 0)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
threshold = 0.2
plt.ion()

print template.shape[::-1]


def callDetect():
    while True:
        image = game.get_game_screen()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None
        cv2.waitKey(1000)
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
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
                print maxVal


        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        if maxVal < 300000:
            print "Not Detected: using best guess"
        # draw a bounding box around the detected result and display the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.imshow("final", image)
        cv2.waitKey(1)
        print "-------"






        # img_rgb = game.get_game_screen()
        # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        # #res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
        # loc = np.where( res >= threshold)
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        #
        # cv2.imshow('title',img_rgb)
        # cv2.waitKey(1)


t = threading.Thread(target=callDetect, args=())
t.deamon = True
t.start()

ScreenWidth = game.get_screen_width()
ScreenHeight = game.get_screen_height()
action = [False, False, False]
actions = [[True, False, False], [False, True, False], [False, False, True]]
# sleep_time = 0.028
sleep_time = 0.028

episodes = 100
avgScore = 0.0
for i in range(episodes):

    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

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
