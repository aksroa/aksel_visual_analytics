#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the packages
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt


# In[2]:


#Defining the path to the image
image1 = os.path.join("..", "Image1.jpg")
#Reading the image
image = cv2.imread(image1)


# In[3]:


#Creating the green rectangle box of the region of interest
cv2.rectangle(image, (1250,875), (2950, 2800), (0, 255, 0), 3) #cv2.rectangle( image, start_point, end_pont, color, thickness)
#Showing the image with ROI
jimshow(image)


# In[4]:


#Saving the cropped image as image_cropped.jpg
cv2.imwrite(os.path.join("..", "data", "image_with_ROI.jpg"), image)


# __Cropping the image__

# In[5]:


#Defining the center of the image
(centerX, centerY) = (image.shape[1]//2, image.shape[0]//2)


# In[6]:


#Cropping the image based on the centers
cropped_image = image[centerY-750:centerY+1150, centerX-750: centerX +700]


# In[7]:


jimshow(cropped_image)


# In[8]:


#Saving the cropped image as image_cropped.jpg
cv2.imwrite(os.path.join("..", "data", "image_cropped.jpg"), cropped_image)


# __Finding every letter in the image__

# In[9]:


#blur image to remove noise
blurred_image = cv2.blur(cropped_image, (3,3))


# In[10]:


jimshow(blurred_image)


# In[11]:


#Flattening the image to black and white
grey = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)


# In[12]:


jimshow_channel(grey)


# In[13]:


# Using canny edge detection
canny = cv2.Canny(grey, 100, 200) #Second and third arguments are our minVal and maxVal.


# In[14]:


jimshow_channel(canny)


# In[15]:


# Using the "findContours" cv2 function to find all the contours from the canny image
    contours, _ =cv2.findContours(canny.copy(), # .copy() is just so that we don't overwrite the original image, but rather do it on a copy
    cv2.RETR_EXTERNAL, # This takes only the external structures (we don't want edges within each coin)
    cv2.CHAIN_APPROX_SIMPLE, ) # The method of getting approximated contours


# In[16]:


# The original cropped image with contour overlay
image_letters =cv2.drawContours(cropped_image.copy(), # draw contours on original
contours, # our list of contours
-1, #which contours to draw -1 means all. -> 1 would mean first contour 
(0, 255, 0), # contour color
2) # contour pixel width


# In[17]:


jimshow(image_letters)


# In[18]:


cv2.imwrite(os.path.join("..", "data", "image_letters.jpg"), image_letters)


# In[ ]:




