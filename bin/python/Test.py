"""
Pyplot animation example.

The method shown here is only for very simple, low-performance
use.  For more demanding applications, look at the animation
module and the examples that use it.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

x = np.arange(6)
y = np.arange(5)
z = x * y[:, np.newaxis]
bool = True
img1 = cv2.imread('HEADB1.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',-1)

p = plt.imshow(img1)
fig = plt.gcf()
plt.title("Boring slide show")
plt.pause(0.5)

for i in range(5):
    if bool == False:
        bool = True
        p.set_data(img1)
    else:
        bool = False
        p.set_data(img2)
    print("step", i)
    plt.pause(0.5)