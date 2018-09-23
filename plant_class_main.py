import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def main():
    labels = []
    minWidth, maxWidth = image_width()
    # Loop through all folders and files, resampling so they all have the same
    # size.
    for folder_name in os.listdir("Data"):
        for file_name in os.listdir("Data/"+folder_name):
            labels.append(folder_name)
            img = Image.open("Data/"+folder_name+"/"+file_name)
            img = img.resize((minWidth,maxWidth))

    print(labels[:10])
    print(labels.shape)


def image_width():
   minWidth = 500
   maxWidth = 500

   for i, folder_name in enumerate(os.listdir("Data")):
        for j, file_name in enumerate(os.listdir("Data/"+folder_name)):
            img = Image.open("Data/"+folder_name+"/"+file_name)
            if np.array(img).shape[0]<minWidth:
                minWidth = np.array(img).shape[0]
            if np.array(img).shape[1]<maxWidth:
                maxWidth = np.array(img).shape[1]

   return minWidth, maxWidth


if __name__=="__main__":
    main()
