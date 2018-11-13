
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
    return np.sum(((hist1 - hist2) >> 6) ** 2) ** 0.5

# Ouverture du stream de lecture de la séquence vidéo
videoCapture = cv2.VideoCapture("pole-vaulter.avi")
videoCapture.open("pole-vaulter.avi")

frame = 0

if videoCapture.isOpened :
    # Lecture de la premiere frame
    reading, image = videoCapture.read()
    prevImage = copy.deepcopy(image)

    seuilCoupure = 2500
    seuilEffet = 445
        
    evaluatingEffect = False
    effectStart = 0
    effectEnd = 0
    effectStartImage = []

    totalTime = time.time()
    while (reading) :
        niveau = 16
        start = time.time()
        histogramR, histogramG, histogramB = histogram(image, niveau)
        if not frame:
            prevhistogramR, prevhistogramG, prevhistogramB = list(histogramR), list(histogramG), list(histogramB)

        distanceR = distance(histogramR, prevhistogramR)
        distanceG = distance(histogramG, prevhistogramG)
        distanceB = distance(histogramB, prevhistogramB)
        #print("Distance R: ", distanceR)
        #print("Distance G: ", distanceG)
        #print("Distance B: ", distanceB)
        totalDistance = 2 * distanceR + distanceG / 2 + distanceB / 2
        #fou seuil trouvé expérimentalement
        if  totalDistance > seuilCoupure:
            evaluatingEffect = False
            print("Coupure à {}, distance de {}".format(frame,totalDistance))
            #print("Distance R: ", distanceR)
            #print("Distance G: ", distanceG)
            #print("Distance B: ", distanceB)
            cv2.imshow("Coupure", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
        elif totalDistance > seuilEffet:
            if not evaluatingEffect:
                evaluatingEffect = True
                effectStart = frame
                effectStartImage = [histogramR,histogramG,histogramB]
                cv2.imshow("EffetStart", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            if evaluatingEffect:
                distanceR = distance(histogramR, effectStartImage[0])
                distanceG = distance(histogramG, effectStartImage[1])
                distanceB = distance(histogramB, effectStartImage[2])
                #print("Distance R: ", distanceR)
                #print("Distance G: ", distanceG)
                #print("Distance B: ", distanceB)
                totalDistance = 2 * distanceR + distanceG / 2 + distanceB / 2
                if  totalDistance > 2300:
                    print("Potentiel début effet à {}, distance de {}".format(effectStart,totalDistance))
                    print("Potentiel fin d'effet à {}, distance de {}".format(frame,totalDistance))
                    cv2.imshow("EffetFin", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                evaluatingEffect = False

        #cv2.imshow("Video", image)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        
        end = time.time()

        #print(frame, " : ", totalDistance)
        #print(frame, end - start)
        frame += 1

        # Lecture de la prochaine frame
        reading, image = videoCapture.read()
        prevhistogramR, prevhistogramG, prevhistogramB = list(histogramR), list(histogramG), list(histogramB)

    endTotal = time.time()

    print(endTotal - totalTime)
    # Fermeture du stream de lecture de la séquence vidéo
    videoCapture.release()
