
''' 
***********************************************************************************************************************************************************************
@File          VideoSequncer.py
@Title         Implementation de la méthode de décomposition d'une séquence vidéo en prises de vue (INF8770 - TP3)
@Author        Vincent Chassé, Pierre To
@Created       31/10/2018
***********************************************************************************************************************************************************************
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import time

# Calcule l'histogramme avec N niveaux
# Il peut y avoir de 1 à 256 niveaux
def histogram(image, niveau) :
    b, g, r = cv2.split(image) # get b, g, r

    histogramR = np.histogram(r, niveau)[0]
    histogramG = np.histogram(g, niveau)[0]
    histogramB = np.histogram(b, niveau)[0]

    return histogramR, histogramG, histogramB

# Calcule la distance euclidienne entre 2 histogrammes
def distance(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2)

# Ouverture du stream de lecture de la séquence vidéo
videoCapture = cv2.VideoCapture("pole-vaulter.avi")
videoCapture.open("pole-vaulter.avi")

if videoCapture.isOpened :
    # Lecture de la premiere frame
    reading, image = videoCapture.read()
    prevImage = copy.deepcopy(image)
    seuilCoupure = 0
    seuilEffet = 0

    compteur = 0
        
    totalTime = time.time()
    while (reading) :
        #print(image)
        #b, g, r = cv2.split(image)      # get b, g, r
        #rgb_image = cv2.merge([r, g, b])  # switch to rgb
        #plt.imshow(rgb_image)
        #plt.show()

        niveau = 128
        start = time.time()
        histogramR, histogramG, histogramB = histogram(image, niveau)
        #print(distance(histogramR, histogramB))
        #print(histogramR)
        end = time.time()

        print(compteur, end - start)
        compteur += 1

        # Lecture de la prochaine frame
        reading, image = videoCapture.read()

    endTotal = time.time()

    print(endTotal - totalTime)
    # Fermeture du stream de lecture de la séquence vidéo
    videoCapture.release()