# Partie 2 – Histogrammes et transformations d’intensité
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Charger l'image lena en niveau de gris ---
lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
if lena is None:
    raise FileNotFoundError("Image lena.jpg introuvable.")

# --- 2. Fonction HISTO : calculer histogramme manuellement ---
def HISTO(img):
    hist = np.zeros(256, dtype=int)  # 256 niveaux de gris
    for pixel in img.flatten():      # Parcourt tous les pixels
        hist[pixel] += 1
    return hist

# --- 3. Afficher histogramme ---
def show_histogram(hist, title):
    plt.figure()
    plt.bar(range(256), hist, width=1)
    plt.title(title)
    plt.xlabel('Niveau de gris')
    plt.ylabel('Nombre de pixels')
    plt.show()

# Histogramme original
hist_lena = HISTO(lena)
show_histogram(hist_lena, "Histogramme de lena (originale)")

# --- 4. Transformation TRL : ajout constant C ---
def TRL(img, C):
    result = img.astype(np.int16) + C  # éviter overflow
    result = np.clip(result, 0, 255)   # limiter entre 0..255
    return result.astype(np.uint8)

# Test TRL
lena_plus50 = TRL(lena, 50)
lena_minus50 = TRL(lena, -50)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(lena_plus50, cmap='gray'); plt.title("TRL +50")
plt.axis('off')
plt.subplot(1,2,2); plt.imshow(lena_minus50, cmap='gray'); plt.title("TRL -50")
plt.axis('off')
plt.show()

# Histogrammes TRL
show_histogram(HISTO(lena_plus50), "Histogramme TRL +50")
show_histogram(HISTO(lena_minus50), "Histogramme TRL -50")

# --- 5. Inversion de l'image ---
lena_inv = 255 - lena

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(lena, cmap='gray'); plt.title("Originale")
plt.axis('off')
plt.subplot(1,2,2); plt.imshow(lena_inv, cmap='gray'); plt.title("Inversée")
plt.axis('off')
plt.show()

# Histogramme inversion
show_histogram(HISTO(lena_inv), "Histogramme après inversion")
