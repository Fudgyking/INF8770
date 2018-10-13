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
            y[i][j] = (((r[i][j]) + 2*(g[i][j]) + (b[i][j])) / 4)
            u[i][j] = (b[i][j]) - (g[i][j])
            v[i][j] = (r[i][j]) - (g[i][j])

    return np.array(y), np.array(u), np.array(v)

# Fonction qui convertit l'espace de couleur YUV vers RGB
def yuv2rgb(y, u, v) :
    r = [[0 for x in range(len(y[0]))] for z in range(len(y))]
    g = [[0 for x in range(len(y[0]))] for z in range(len(y))]
    b = [[0 for x in range(len(y[0]))] for z in range(len(y))]

    for i in range(len(y)) :
        for j in range(len(y[0])) :
            g[i][j] = y[i][j] - ((u[i][j] + v[i][j])/4)
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

        for column in range(len(u[0])) :
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

# Fonction qui applique la transformée en ondelettes discrète de Haar 
# avec un niveau de récursion défini sur le canal x
# l : low, h : high
def DWT(x, nbRecursion) :
    if nbRecursion == 0 :
        return x

    # En X : filtrage passe-bas. Enlève les hautes fréquences en faisant une moyenne
    xl = (x[:,::2] + x[:,1::2])/2

    # En X : filtrage passe-haut. On extrait les hautes fréquences à l'aide des différences.
    xh = (x[:,::2] - x[:,1::2])/2

    # Poursuivre le traitement selon Y des images x1 et xh
    # Sous-bande 1 : Basses fréquences en X et Y
    xll = (xl[::2,:] + xl[1::2,:])/2

    # Sous-bande 2 : Hautes fréquences en Y, basses en X
    xlh = (xl[::2,:] - xl[1::2,:])/2

    # Sous-bande 3 : Hautes fréquences en X, basses en Y
    xhl = (xh[::2,:] + xh[1::2,:])/2

    # Sous-bande 4 : Hautes fréquences en X et Y
    xhh = (xh[::2,:] - xh[1::2,:])/2

    # Appel récursif
    xll = DWT(xll, nbRecursion - 1)

    x = np.concatenate(( np.concatenate((xll, xhl)), np.concatenate((xlh, xhh))), axis = 1 )

    return np.array(x)

# Fonction qui applique la transformée inverse en ondelettes discrète de Haar 
# avec un niveau de récursion défini sur le canal x
def iDWT(x, nbRecursion) :
    if (nbRecursion == 0) :
        return x

    # XL
    # augmenter la taille de l'image et ajout des hautes fréquences
    xll = x[:int(len(x) / (2**nbRecursion)), :int(len(x[0]) / (2**nbRecursion))]
    xlh = x[:int(len(x) / (2**nbRecursion)), int(len(x[0]) / (2**nbRecursion)):]

    # copie des pixels de xl. Met les mêmes valeurs de xll pour i et i + 1
    xl = np.zeros((len(xll) * 2, len(xll[0])))
    for i in range(len(xll)) :
        for j in range(len(xll[0])) :
            xl[2 * i, j] = xll[i, j] + xlh[i, j]
            xl[2 * i + 1, j] = xll[i, j] - xlh[i, j]

    # XH
    # augmenter la taille de l'image et ajout des hautes fréquences
    xhl = x[int(len(x)/(2**nbRecursion)):, :int(len(x[0])/(2**nbRecursion))]
    xhh = x[int(len(x)/(2**nbRecursion)):, int(len(x[0])/(2**nbRecursion)):]

    # copie des pixels de xh. Met les mêmes valeurs de xhl pour i et i + 1
    xh = np.zeros((len(xhl) * 2, len(xhl[0])))
    for i in range(len(xhl)) :
        for j in range(len(xhl[0])) :
            xh[2 * i, j] = xhl[i, j] + xhh[i, j]
            xh[2 * i + 1, j] = xhl[i, j] - xhh[i, j]

    # X
    # copie des pixels de xl et xh.
    xRes = np.zeros((len(x), len(x[0])))
    for i in range(len(xl)) :
        for j in range(len(xl[0])) :
            xRes[i, j * 2] = xl[i, j] + xh[i, j]
            xRes[i, j * 2 + 1] = xl[i, j] - xh[i, j]

    return np.array(xRes)

# Ce script effectue la conversion de l'espace de couleur d'une image RGB vers YUV (réversible).
# Nous utilisons un sous-échantillonnage 4:2:0.
# Nous utilisons les équations suivantes :
# Y = (R + 2G + B) / 4, U = B - G, V = R - G
# R = V + G, G = Y - (U+V) / 4, B = U + G

# Lecture de l'image originale
image = (cv2.imread('img/image5.jpg')).astype(float)
b, g, r = cv2.split(image)      # get b, g, r
#rgb_image = cv2.merge([r,g,b])  # switch to rgb
#plt.imshow(rgb_image)
#plt.show()

# mettre toutes les valeurs flottantes entre 0 et 1
b, g, r = [x/255 for x in [b, g, r]]

# conversion de l'espace des couleurs RGB vers YUV
y, u, v = rgb2yuv(r, g, b)

# sous-échantillonnage 4:2:0
u, v = subSampling(4, 2, 0, u, v)

# transformée en ondelettes discrète de Haar (avec trois étages)
nbRecursion = 1
y = DWT(y, nbRecursion)
u = DWT(u, nbRecursion)
v = DWT(v, nbRecursion)

# transformée inverse en ondelettes discrète de Haar (avec trois étages)
y = iDWT(y, nbRecursion)
u = iDWT(u, nbRecursion)
v = iDWT(v, nbRecursion)

# conversion de YUV vers RGB
newR, newG, newB = yuv2rgb(y, u, v)

#print([ [newR[x][y] - r[x][y] for x in range(len(newR[0]))] for y in range(len(newR)) ])
rgb_image = cv2.merge([newR,newG,newB])  # switch to rgb
plt.imshow(rgb_image)
plt.show()