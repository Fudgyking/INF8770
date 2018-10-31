
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

    #histogramR = np.zeros(niveau)
    #histogramG = np.zeros(niveau)
    #histogramB = np.zeros(niveau)

    r = r.ravel()
    g = g.ravel()
    b = b.ravel()
    
    histogramR = [0 for i in range(niveau)] 
    histogramG = [0 for i in range(niveau)] 
    histogramB = [0 for i in range(niveau)]
    
    pas = int(256 / niveau)

    for i in range(len(r)) :
        #for j in range(len(r[0])) :
            histogramR[int(r[i]/pas)] += 1
            histogramG[int(g[i]/pas)] += 1
            histogramB[int(b[i]/pas)] += 1

    return histogramR, histogramG, histogramB

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
    
    while (reading) :
        #print(image)
        #b, g, r = cv2.split(image)      # get b, g, r
        #rgb_image = cv2.merge([r, g, b])  # switch to rgb
        #plt.imshow(rgb_image)
        #plt.show()

        niveau = 256
        start = time.time()
        histogramR, histogramG, histogramB = histogram(image, niveau)
        end = time.time()

        print(compteur, end - start)
        compteur += 1

        # Lecture de la prochaine frame
        reading, image = videoCapture.read()

    # Fermeture du stream de lecture de la séquence vidéo
    videoCapture.release()