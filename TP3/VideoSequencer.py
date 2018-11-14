
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

    seuilCoupure = 2500
    seuilEffet = 445
        
    evaluatingEffect = False
    isFondu = False
    frameEffect = 0
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
                frameEffect = 0
                effectStart = frame
                effectStartImage = [histogramR,histogramG,histogramB]
                # cv2.imshow("EffetFin", image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        else:
            if evaluatingEffect:
                distanceR = distance(histogramR, effectStartImage[0])
                distanceG = distance(histogramG, effectStartImage[1])
                distanceB = distance(histogramB, effectStartImage[2])

                totalDistance = 2 * distanceR + distanceG / 2 + distanceB / 2
                if totalDistance > 2300:
                    print("Début effet à {}, distance de {}".format(effectStart,totalDistance))
                    print("Fin d'effet à {}, distance de {}".format(frame,totalDistance))
                    # cv2.imshow("EffetFin", image)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                evaluatingEffect = False

        #grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #grayImageBlur = cv2.GaussianBlur(grayImage,(5,5),3)
        #cv2.imshow("Gray", grayImage)
        #diffGauss = np.absolute(grayImage - grayImageBlur)
        #diffGauss = diffGauss * 255/np.max(diffGauss)
        #cv2.imshow("GrayBlur", diffGauss)
        
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

        # calcul de la convolution avec les filtres de Sobel selon l'axe X et l'axe Y
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
        cv2.imshow("Dilated Edges", dilatedEdges)

        if not frame:
            prevEdges, prevDilatedEdges = list(edges), list(dilatedEdges)
        
        # Aretes entrantes
        edgesIn = 1 - (np.sum(prevDilatedEdges * edges) / np.sum(edges))

        # Aretes sortants
        edgesOut = 1 - (np.sum(prevEdges * dilatedEdges) / np.sum(prevEdges))

        #print("EdgesIn:", edgesIn)
        #print("EdgesOut:", edgesOut)

        # Détection d'une coupure
        if max(edgesIn, edgesOut) > -50:
            print("Coupure EDGES à {}, distance de {}".format(frame, max(edgesIn, edgesOut)))

        # Détection d'un fondu
        if max(edgesIn, edgesOut) > -100:
            frameEffect = frameEffect + 1
        else:
            if frameEffect > 3:
                print("Fin d'effet à {}, frameEffect de {}".format(frame, frameEffect))
            frameEffect = 0

        # if not isFondu:
        #     fondu = edgesIn > edgesOut
        # else:
        #     fondu = edgesIn < edgesOut

        # if frameEffect == 0:
        #     prevFondu = fondu

        # if fondu == prevFondu:
        #     frameEffect = frameEffect + 1
        # else:
        #     if frameEffect > 4:
        #         if not isFondu:
        #             print("Début effet à {}, frameEffect de {}".format(frame - frameEffect, frameEffect))
        #             isFondu = True
        #         else:
        #             print("Fin d'effet à {}, frameEffect de {}".format(frame, frameEffect))
        #             isFondu = False
        #     else:
        #         isFondu = False
            
        #     frameEffect = 0
        
        # prevFondu = fondu
        
        #if frame == 589:
        #    cv2.imshow("589", dilatedEdges)

        #if frame == 590:
        #    print(max(edgesIn, edgesOut))
        #    cv2.imshow("590", dilatedEdges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end = time.time()

        #print(frame, " : ", totalDistance)
        #print(frame, end - start)
        frame += 1

        # Lecture de la prochaine frame
        reading, image = videoCapture.read()
        prevhistogramR, prevhistogramG, prevhistogramB = list(histogramR), list(histogramG), list(histogramB)
        prevEdges, prevDilatedEdges = list(edges), list(dilatedEdges)

    endTotal = time.time()

    print(endTotal - totalTime)
    # Fermeture du stream de lecture de la séquence vidéo
    videoCapture.release()
