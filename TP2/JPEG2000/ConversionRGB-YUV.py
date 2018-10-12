import numpy as np
import matplotlib.pyplot as plt
import cv2

# Fonction qui convertit l'espace de couleur RGB vers YUV
def rgb2yuv(r, g, b) :
    y = [[0 for x in range(len(r[0]))] for y in range(len(r))]
    u = [[0 for x in range(len(r[0]))] for y in range(len(r))]
    v = [[0 for x in range(len(r[0]))] for y in range(len(r))]

    for i in range(len(r)) :
        for j in range(len(r[0])) :
            y[i][j] = int((int(r[i][j]) + 2*int(g[i][j]) + int(b[i][j])) / 4)
            u[i][j] = int(b[i][j]) - int(g[i][j])
            v[i][j] = int(r[i][j]) - int(g[i][j])

    return np.array(y), np.array(u), np.array(v)

# Fonction qui convertit l'espace de couleur YUV vers RGB
def yuv2rgb(y, u, v) :
    r = [[0 for x in range(len(y[0]))] for z in range(len(y))]
    g = [[0 for x in range(len(y[0]))] for z in range(len(y))]
    b = [[0 for x in range(len(y[0]))] for z in range(len(y))]

    for i in range(len(y)) :
        for j in range(len(y[0])) :
            g[i][j] = y[i][j] - int((u[i][j] + v[i][j])/4)
            r[i][j] = v[i][j] + g[i][j]
            b[i][j] = u[i][j] + g[i][j]

    return np.array(r), np.array(g), np.array(b)

# Fonction qui effectue un sous-échantillonnage de la chrominance 
# avec un ratio J:a:b de paramètres U (Cb), V (Cr)
def subSampling(j, a, b, u, v) :
    for line in range(0, len(u), 2) :
        onlyLineA = False

        if line + 1 >= len(u) :
            onlyLineA = True

        cbLineA = u[line]
        crLineA = v[line]

        if not onlyLineA :
            crLineB = v[line + 1]
            cbLineB = u[line + 1]

        for column in range(0, len(u[0])) :
            # a : nb d'échantillons Cr et Cb dans la première rangée de j pixels (j >= a > 0)
            sampleALine = int(j / a)
            index = int(column / sampleALine) * sampleALine
            cbLineA[column] = cbLineA[index]
            crLineA[column] = crLineA[index]

            # b : nb d'échantillons Cr et Cb dans la deuxième rangée de j pixels (j >= b >= 0)
            if not onlyLineA :
                if (b > 0) :
                    sampleBLine = int(j / b)
                    index = int(column / sampleBLine) * sampleBLine
                    cbLineB[column] = cbLineB[index]
                    crLineB[column] = crLineB[index]
                else : # b = 0
                    cbLineB[column] = cbLineA[column]
                    crLineB[column] = crLineA[column]

    return np.array(u), np.array(v)

# Ce script effectue la conversion de l'espace de couleur d'une image RGB vers YUV (réversible).
# Nous utilisons un sous-échantillonnage 4:2:0.
# Nous utilisons les équations suivantes :
# Y = (R + 2G + B) / 4, U = B - G, V = R - G
# R = V + G, G = Y - (U+V) / 4, B = U + G

# Lecture de l'image originale
image = (cv2.imread('image4.jpg'))
b, g, r = cv2.split(image)      # get b, g, r
#rgb_image = cv2.merge([r,g,b])  # switch to rgb
#plt.imshow(rgb_image)
#plt.show()

# conversion de l'espace des couleurs RGB vers YUV
y, u, v = rgb2yuv(r, g, b)

# sous-échantillonnage 4:2:0
u, v = subSampling(4, 2, 0, u, v)

# conversion de YUV vers RGB
newR, newG, newB = yuv2rgb(y, u, v)

#print([ [newR[x][y] - r[x][y] for x in range(len(newR[0]))] for y in range(len(newR)) ])
rgb_image = cv2.merge([newR,newG,newB])  # switch to rgb
plt.imshow(rgb_image)
plt.show()