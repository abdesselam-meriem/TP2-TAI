# Partie 3 – Expansion (étirement) d’histogramme
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image alex en niveaux de gris
alex = cv2.imread('alex.png', cv2.IMREAD_GRAYSCALE)
if alex is None:
    raise FileNotFoundError("alex.png introuvable dans le dossier !")

# Fonction histogramme (réutilisation TP2 part2)
def HISTO(img):
    hist = np.zeros(256, dtype=int)
    for p in img.flatten():
        hist[p] += 1
    return hist

# 1. Trouver Imin et Imax
Imin = np.min(alex)
Imax = np.max(alex)

print("Valeur minimale:", Imin)
print("Valeur maximale:", Imax)

# 2. Expansion d'histogramme
def expand_contrast(img):
    Imin = np.min(img)
    Imax = np.max(img)
    expanded = (img - Imin) * (255 / (Imax - Imin))
    return expanded.astype(np.uint8)

alex_expanded = expand_contrast(alex)

# 3. Affichage
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(alex, cmap='gray'); plt.title("alex - Originale"); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(alex_expanded, cmap='gray'); plt.title("Après expansion de contraste"); plt.axis('off')
plt.show()

# Hist avant/après
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.bar(range(256), HISTO(alex)); plt.title("Histogramme avant expansion")
plt.subplot(1,2,2); plt.bar(range(256), HISTO(alex_expanded)); plt.title("Histogramme après expansion")
plt.show()
