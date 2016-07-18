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
#game.set_screen_format(ScreenFormat.CRCGCB)
game.init()
print "Doom initialized."


#img = cv2.imread('messi5.jpg')
img = game.get_game_screen()
print img
print img.shape
img = cv2.imread('HEADA1.png',0)
print img.shape
#img = cv2.cvtColor(img, img, cv2.COLOR_RGBA2RGB)
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()

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
