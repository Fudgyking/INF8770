
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

    # Variables pour la méthode basée sur les histogrammes
    niveau = 16
    seuilCoupure = 2500
    seuilEffet = 445
    evaluatingEffect = False
    effectStart = 0
    effectStartImage = []

    totalTime = time.time()
    
    while (reading) :
        # Affichage de la vidéo
        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Méthode basée sur les histogrammes
        histogramR, histogramG, histogramB = histogram(image, niveau)

        if not frame:
            prevhistogramR, prevhistogramG, prevhistogramB = list(histogramR), list(histogramG), list(histogramB)

        distanceR = distance(histogramR, prevhistogramR)
        distanceG = distance(histogramG, prevhistogramG)
        distanceB = distance(histogramB, prevhistogramB)

        totalDistance = 2 * distanceR + distanceG / 2 + distanceB / 2
        
        if  totalDistance > seuilCoupure:
            evaluatingEffect = False
            print("Coupure à {}, distance de {}".format(frame,totalDistance))
            # cv2.imshow("Coupure", image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        elif totalDistance > seuilEffet:
            if not evaluatingEffect:
                evaluatingEffect = True
                effectStart = frame
                effectStartImage = [histogramR,histogramG,histogramB]
                # cv2.imshow("Effet Début", image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        else:
            if evaluatingEffect:
                distanceR = distance(histogramR, effectStartImage[0])
                distanceG = distance(histogramG, effectStartImage[1])
                distanceB = distance(histogramB, effectStartImage[2])

                totalDistance = 2 * distanceR + distanceG / 2 - 2 * distanceB
                if totalDistance > 600:
                    print("Début effet à {}, distance de {}".format(effectStart,totalDistance))
                    print("Fin d'effet à {}, distance de {}".format(frame,totalDistance))
                    # cv2.imshow("Effet Fin", image)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                evaluatingEffect = False
        
        # Décomposition par comparaison des arêtes

        # Calcul de gradients et d'extraction d'arêtes
        # Calcul des gradients sur une image est toujours sur l'image d'intensité
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Pour faire la convolution avec un filtre de Sobel, il faut ajouter des lignes et colonnes additionnelles 
        # pour permettre les calculs aux frontières. Ici, on duplique les valeurs des frontières.
        col = grayImage[:, 0]
        grayImage = np.column_stack((col, grayImage))
        col = grayImage[:, len(grayImage[0])-1]
        grayImage = np.column_stack((grayImage, col))
        row = grayImage[0, :]
        grayImage = np.row_stack((row,grayImage))
        row = grayImage[len(grayImage)-1, :]
        grayImage = np.row_stack((grayImage, row))

        # Calcul de la convolution avec les filtres de Sobel selon l'axe X et l'axe Y
        sobelx = cv2.Sobel(grayImage, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(grayImage, cv2.CV_64F, 0, 1, ksize=3)

        # Extraction d'arêtes à partir des gradients. On calcule la force des gradients, et ensuite on seuille la force des gradients calculée.
        ForceGradient = np.sqrt(np.power(sobelx,2) + np.power(sobely,2))
        thresh = 80
        edges = cv2.threshold(ForceGradient, thresh, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow("Edges", edges)

        # Dilatation des arêtes
        kernel = np.ones((3, 3), np.uint8)
        dilatedEdges = cv2.dilate(edges, kernel, iterations = 1)
        # cv2.imshow("Dilated Edges", dilatedEdges)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        if not frame:
            prevEdges, prevDilatedEdges = list(edges), list(dilatedEdges)
        
        # Aretes entrantes
        edgesIn = 1 - (np.sum(prevDilatedEdges * edges) / np.sum(edges))

        # Aretes sortants
        edgesOut = 1 - (np.sum(prevEdges * dilatedEdges) / np.sum(prevEdges))

        # Détection d'une coupure
        if max(edgesIn, edgesOut) > -50:
            print("Coupure à {}, distance de {} (comparaison des aretes)".format(frame, max(edgesIn, edgesOut)))

        # Lecture de la prochaine frame
        frame += 1
        reading, image = videoCapture.read()
        prevhistogramR, prevhistogramG, prevhistogramB = list(histogramR), list(histogramG), list(histogramB)
        prevEdges, prevDilatedEdges = list(edges), list(dilatedEdges)

    # Temps d'exécution
    endTotal = time.time()
    print("Temps d'exécution : {}".format(endTotal - totalTime))

    # Fermeture du stream de lecture de la séquence vidéo
    videoCapture.release()
