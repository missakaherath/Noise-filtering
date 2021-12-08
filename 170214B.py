# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:31:31 2021

@author: Missaka
"""
import cv2 as cv
import numpy as np
import os

# change the size of the kernel by changing the value of N, kernel is an N*N matrix
N = 3

lenMask = N
mask = []
for i in range(lenMask):
    mask.append([1]*lenMask)

def readimg(path):  #read all the jpg and jpeg files in the given path
    names = []
    matrices = []
    
    for filename in os.listdir(path):
        if (filename.endswith(".jpeg")|filename.endswith(".jpg")):
            img = cv.imread(os.path.join(path,filename))
            if img is not None:
                names.append(filename)
                matrices.append(img)
    return names,matrices

imgnames, imgmatrices = readimg('./')   # read files

def splitChannels(imgMatrix):   # split image into 3 channels
    
    splittedImgR, splittedImgG, splittedImgB = [],[],[]
    
    for i in (imgMatrix):   #iterate through rows
        rowR = []
        rowB = []
        rowG = []
        for j in i: #iterate through pixels
          rowR.append(j[0])
          rowG.append(j[1])
          rowB.append(j[2])
        splittedImgR.append(rowR) 
        splittedImgG.append(rowG) 
        splittedImgB.append(rowB) 
    print()
    
    return splittedImgR, splittedImgG, splittedImgB

def wrap(img):  # wrap around the edge pixels
    img.insert(0, img[-1])
    img.append(img[1])
    for a in range (len(img)-2):
        img[a].insert(0, img[a][-1])
        img[a].append(img[a][1])
    return(img)

def meanFilter(img, mask):
    offset = len(mask)//2
    outputImg = []
    
    for i in range(offset, len(img)-offset):
        outputRow = []
        for j in range(offset, len(img[0])-offset):
            val = 0
            for x in range(len(mask)):
                for y in range(len(mask)):
                    xn = i+x-offset
                    yn = j+y-offset
                    val += (img[xn][yn] * mask[x][y])
            outputRow.append(val/(lenMask**2))
        outputImg.append(outputRow)
    return outputImg

def medianFilter(img, mask):
    offset = len(mask)//2
    outputImg = []
    
    for i in range(offset, len(img)-offset):
        outputRow = []
        for j in range(offset, len(img[0])-offset):
            val = 0
            valarr = []
            for x in range(len(mask)):
                for y in range(len(mask)):
                    xn = i+x-offset
                    yn = j+y-offset
                    val = img[xn][yn]
                    valarr.append(val)
            valarr.sort()
            outputRow.append(valarr[(len(valarr)-1)//2])
        outputImg.append(outputRow)
    return outputImg

def midPointFilter(img, mask):
    offset = len(mask)//2
    outputImg = []
    
    for i in range(offset, len(img)-offset):
        outputRow = []
        for j in range(offset, len(img[0])-offset):
            val = 0
            valarr = []
            for x in range(len(mask)):
                for y in range(len(mask)):
                    xn = i+x-offset
                    yn = j+y-offset
                    val = img[xn][yn]
                    valarr.append(val)
            valarr.sort()
            output = ((int(valarr[0])+int(valarr[-1])))/2
            outputRow.append(output)
        outputImg.append(outputRow)
    return outputImg

        
for i in range (len(imgmatrices)):  #iterate through images 
    splittedImgR, splittedImgG, splittedImgB = splitChannels(imgmatrices[i])    # split into 3 channels
    wrappedR = wrap(splittedImgR)
    wrappedG = wrap(splittedImgG)
    wrappedB = wrap(splittedImgB)
    
    #apply mean filter
    print("Applying Mean filter to :",imgnames[i])
    print()
    meanR = meanFilter(wrappedR, mask)
    meanG = meanFilter(wrappedG, mask)
    meanB = meanFilter(wrappedB, mask)
    
    mergedMean = []
    
    for r in range(len(meanR)):
        mergedMeanRow = []
        for c in range(len(meanR[0])):
            mergedPixel = []
            mergedPixel.append(meanR[r][c])
            mergedPixel.append(meanG[r][c])
            mergedPixel.append(meanB[r][c])
            mergedMeanRow.append(mergedPixel)
        mergedMean.append(mergedMeanRow)
        
    # save mean filtered image
    imagefilename = imgnames[i] + " - Mean" +  ".jpg"
    nparray = np.array(mergedMean)
    cv.imwrite(imagefilename, nparray)

    #apply median filter
    print("Applying Median filter to :",imgnames[i])
    print()
    medianR = medianFilter(wrappedR, mask)
    medianG = medianFilter(wrappedG, mask)
    medianB = medianFilter(wrappedB, mask)
    
    mergedMedian = []
    
    for r in range(len(medianR)):
        mergedMedianRow = []
        for c in range(len(medianR[0])):
            mergedPixel = []
            mergedPixel.append(meanR[r][c])
            mergedPixel.append(meanG[r][c])
            mergedPixel.append(meanB[r][c])
            mergedMedianRow.append(mergedPixel)
        mergedMedian.append(mergedMedianRow)
        
    # save median filtered image
    imagefilename = imgnames[i] + " - Median" + ".jpg"
    nparray = np.array(mergedMedian)
    cv.imwrite(imagefilename, nparray)
    
    
    #apply Mid-point filter
    print("Applying Mid-point filter to :",imgnames[i])
    print()
    midPointR = midPointFilter(wrappedR, mask)
    midPointG = midPointFilter(wrappedG, mask)
    midPointB = midPointFilter(wrappedB, mask)
    
    mergedMidPoint = []
    
    for r in range(len(midPointR)):
        mergedMidPointRow = []
        for c in range(len(midPointR[0])):
            mergedPixel = []
            mergedPixel.append(midPointR[r][c])
            mergedPixel.append(midPointG[r][c])
            mergedPixel.append(midPointB[r][c])
            mergedMidPointRow.append(mergedPixel)
        mergedMidPoint.append(mergedMidPointRow)
        
    # save mid point filtered image
    imagefilename = imgnames[i] + " - Mid point" +  ".jpg"
    nparray = np.array(mergedMidPoint)
    cv.imwrite(imagefilename, nparray)