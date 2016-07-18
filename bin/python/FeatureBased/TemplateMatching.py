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

#doom stuff
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

template = cv2.imread('HEADA2.png',0)
template = cv2.Canny(template, 50, 200)
h, w = template.shape[::-1]
threshold = 0.2
plt.ion()

print template.shape[::-1]

def callDetect():
    while True: 
        img_rgb = game.get_game_screen()
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        #res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        cv2.imshow('title',img_rgb)
        cv2.waitKey(1)

t = threading.Thread(target=callDetect, args=())
t.deamon = True
t.start()


ScreenWidth = game.get_screen_width()
ScreenHeight = game.get_screen_height()
action = [False, False, False]
actions = [[True,False,False],[False,True,False],[False,False,True]]
#sleep_time = 0.028
sleep_time = 0.028

episodes = 100
avgScore = 0.0
for i in range(episodes):
    
    print("Episode #" + str(i+1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()


    while not game.is_episode_finished():

        # Gets the state
        s = game.get_state()

        # Makes a random action and get remember reward.
        r = game.make_action(action, 1)
        #r = game.make_action(choice(actions), 1)
        # Prints state's game variables. Printing the image is quite pointless.
       # print("State #" + str(s.number))
       # print("Game variables:", s.game_variables[0])
       # print("Reward:", r)
       # print("=====================")

        if sleep_time>0:
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