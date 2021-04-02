#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:46:00 2021

@author: ubuntu
"""
#source : https://developers.google.com/codelabs/tensorflow-3-convolutions#1
#https://developers.google.com/codelabs/tensorflow-3-convolutions#2

import cv2
import numpy as np
from scipy import misc

img = misc.ascent()
import matplotlib.pyplot as plt

plt.grid(False)
#plt.gray()
plt.axis('off')
plt.imshow(img)
plt.show()

transformed_img = np.copy(img)
#print(img)
#print("\n")
#print(transformed_img)
size_x = transformed_img.shape[0]
size_y = transformed_img.shape[1]
print(transformed_img.shape)

# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.
filter = [[-1,-2,-1],[0,0,0], [1,2,1]]
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
print(filter)
filt = np.transpose(filter)
print(filt)     #try using filter transpose
print("filter element: ", filter[0][0] )
print("filter element: ", filter[0][1] )
print("filter element: ", filter[1][0] )
print("filter element: ", filter[2][0] )


# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them

weight = 1

#https://developers.google.com/codelabs/tensorflow-3-convolutions#3
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0.0
        convolution = convolution + (img[x-1,y-1] * filter[0][0])
        convolution = convolution + (img[x, y-1]) * filt[0][1]
        convolution = convolution + (img [x +1, y-1]) * filt[0][2]
        convolution = convolution + (img[x-1, y] * filt[1][0])
        convolution = convolution + (img[x, y] * filt[1][1])
        convolution = convolution + (img[x+1, y] * filt[1][2])
        convolution = convolution + (img[x-1, y+1] * filt[2][0])
        convolution = convolution + (img[x, y+1] * filt[2][1])
        convolution = convolution + (img[x+1, y+1] * filt[2][2])
        
        convolution = convolution + weight
        if convolution < 0:
            convolution = 0
        if convolution > 255:
            convolution = 255
        transformed_img[x, y] = convolution
        
#plot the image. Note the size of axes , 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(transformed_img)
plt.axis(False)
plt.show()

#Pooling    
#Similar to convolutions, pooling greatly helps with detecting features. 
#Pooling layers reduce the overall amount of information in an image while maintaining the features that are detected as present.       
        
#MAX pooling - The idea here is to iterate over the image, and look at the pixel 
#and it's immediate neighbors to the right, beneath, and right-beneath.
#Take the largest (hence the name MAX pooling) of them and load it into the new image. 
#Thus the new image will be 1/4 the size of the old 
        
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(transformed_img[x, y])
    pixels.append(transformed_img[x+1, y])
    pixels.append(transformed_img[x, y+1])
    pixels.append(transformed_img[x+1, y+1])
    pixels.sort(reverse=True)
    newImage[int(x/2),int(y/2)] = pixels[0]

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.axis('off')
plt.show() 

#This code will show a (2, 2) pooling.Run it to see the output,
#and you'll see that while the image is 1/4 the size of the original, the extracted features are maintained!