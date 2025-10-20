# Partie 1 du TP2 - Manipulations arithmétiques & logiques (version en niveaux de gris)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def imshow_gray(title, img):
    if img is None:
        print(f"{title} : image None")
        return
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')

# 1) Charger l'image lena (grayscale)
lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
if lena is None:
    raise FileNotFoundError("Impossible de charger 'lena.jpg' — vérifie qu'elle est dans le même dossier.")

h, w = lena.shape

# 2) Créer une image binaire B (noir) puis dessiner un rectangle blanc positionné aléatoirement
B = np.zeros((h, w), dtype=np.uint8)

# Générer rectangle aléatoire
rect_w = random.randint(int(w * 0.1), int(w * 0.5))
rect_h = random.randint(int(h * 0.1), int(h * 0.5))
x0 = random.randint(0, w - rect_w)
y0 = random.randint(0, h - rect_h)
x1, y1 = x0 + rect_w, y0 + rect_h

cv2.rectangle(B, (x0, y0), (x1, y1), color=255, thickness=-1)

# 3) Opérations arithmétiques (grayscale)
add_img = cv2.add(lena, B)        # addition saturée
sub_img = cv2.subtract(lena, B)   # soustraction saturée
mul_img = cv2.multiply(lena, B // 255)  # multiplication masque binaire (0 ou 1)

# 4) Opérations logiques (bitwise)
and_img = cv2.bitwise_and(lena, B)
or_img  = cv2.bitwise_or(lena, B)
xor_img = cv2.bitwise_xor(lena, B)

# 5) Affichage avec matplotlib
plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1); imshow_gray('lena (grayscale)', lena)
plt.subplot(3, 3, 2); imshow_gray('B (binaire rectangle)', B)
plt.subplot(3, 3, 3); imshow_gray('Addition (lena + B)', add_img)
plt.subplot(3, 3, 4); imshow_gray('Soustraction (lena - B)', sub_img)
plt.subplot(3, 3, 5); imshow_gray('Multiplication (lena * Bmask)', mul_img)
plt.subplot(3, 3, 6); imshow_gray('AND bitwise', and_img)
plt.subplot(3, 3, 7); imshow_gray('OR bitwise', or_img)
plt.subplot(3, 3, 8); imshow_gray('XOR bitwise', xor_img)

plt.tight_layout()
plt.show()

print("Partie 1 terminée — affichage des images en niveaux de gris. Rectangle :", (x0, y0, x1, y1))
