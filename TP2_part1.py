# TAI_TP2_part1.py
# Partie 1 du TP2 - Manipulations arithmétiques & logiques
# Prérequis : cv2, numpy, matplotlib
# Exécuter dans le même dossier que lena.jpg

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def imshow_cv_rgb(title, img):
    """
    Affiche une image BGR (OpenCV) correctement en RGB dans matplotlib.
    """
    if img is None:
        print(f"{title} : image None")
        return
    # Si image est grayscale (2D), matplotlib l'affichera en cmap='gray'
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        # Convertir BGR -> RGB
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

# 1) Charger l'image lena (couleur)
lena = cv2.imread('lena.jpg')
if lena is None:
    raise FileNotFoundError("Impossible de charger 'lena.jpg' — vérifie qu'elle est dans le même dossier.")

h, w = lena.shape[:2]

# 2) Créer une image binaire B (noir) puis dessiner un rectangle blanc positionné aléatoirement
B = np.zeros((h, w), dtype=np.uint8)

# Générer rectangle aléatoire (assurer que la taille > 0)
rect_w = random.randint(int(w*0.1), int(w*0.5))
rect_h = random.randint(int(h*0.1), int(h*0.5))
x0 = random.randint(0, w - rect_w)
y0 = random.randint(0, h - rect_h)
x1, y1 = x0 + rect_w, y0 + rect_h

# Dessiner rectangle blanc (valeur 255) sur B
cv2.rectangle(B, (x0, y0), (x1, y1), color=255, thickness=-1)

# Pour opérer avec lena (3 canaux), on convertit B -> 3 canaux (uint8)
B_color = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)

# 3) Opérations arithmétiques
# Important : OpenCV effectue clip automatique pour add/sub si on utilise cv2.add/sub,
# mais np.add/np.subtract peuvent overflow si on ne convertit pas correctement.
add_img = cv2.add(lena, B_color)        # addition saturée
sub_img = cv2.subtract(lena, B_color)   # soustraction saturée
mul_img = cv2.multiply(lena, B_color // 255)  # on multiplie par 1 ou 0 (B_color/255)

# 4) Opérations logiques (bitwise) — travailler sur uint8 (chaînes BGR)
and_img = cv2.bitwise_and(lena, B_color)
or_img  = cv2.bitwise_or(lena, B_color)
xor_img = cv2.bitwise_xor(lena, B_color)

# 5) Affichage compact avec matplotlib
plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1); imshow_cv_rgb('lena (origine)', lena)
plt.subplot(3, 3, 2); imshow_cv_rgb('B (binaire rectangle)', B)            # affichage grayscale
plt.subplot(3, 3, 3); imshow_cv_rgb('B (en RGB)', B_color)

plt.subplot(3, 3, 4); imshow_cv_rgb('Addition (lena + B)', add_img)
plt.subplot(3, 3, 5); imshow_cv_rgb('Soustraction (lena - B)', sub_img)
plt.subplot(3, 3, 6); imshow_cv_rgb('Multiplication (lena * Bmask)', mul_img)

plt.subplot(3, 3, 7); imshow_cv_rgb('AND bitwise', and_img)
plt.subplot(3, 3, 8); imshow_cv_rgb('OR bitwise', or_img)
plt.subplot(3, 3, 9); imshow_cv_rgb('XOR bitwise', xor_img)

plt.tight_layout()
plt.show()

# 6) Sauvegarde optionnelle (décommenter si tu veux enregistrer)
# cv2.imwrite('part1_add.png', add_img)
# cv2.imwrite('part1_sub.png', sub_img)
# cv2.imwrite('part1_mul.png', mul_img)
# cv2.imwrite('part1_and.png', and_img)
# cv2.imwrite('part1_or.png', or_img)
# cv2.imwrite('part1_xor.png', xor_img)

print("Partie 1 terminée — affichage des images. Rectangle utilisé :", (x0,y0,x1,y1))
