# Partie 4 – Seuillage manuel et seuillage automatique (Otsu)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger lena et alex en niveaux de gris
lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
alex = cv2.imread('alex.png', cv2.IMREAD_GRAYSCALE)

if lena is None or alex is None:
    raise FileNotFoundError("Vérifiez que lena.jpg et alex.png sont dans le dossier.")

# --- 1. Seuillage simple (manuel) ---
def seuil_manuel(img, seuil):
    # Appliquer : pixel >= seuil → 255 sinon 0
    _, binary = cv2.threshold(img, seuil, 255, cv2.THRESH_BINARY)
    return binary

# Tests : seuil manuel sur lena et alex
lena_thresh_120 = seuil_manuel(lena, 120)
alex_thresh_120 = seuil_manuel(alex, 120)

# --- 2. Otsu automatique ---
def seuil_otsu(img):
    _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_otsu

lena_otsu = seuil_otsu(lena)
alex_otsu = seuil_otsu(alex)

# --- 3. Affichage résultats ---
plt.figure(figsize=(12,6))

# Lena
plt.subplot(2,3,1); plt.imshow(lena, cmap='gray'); plt.title("Lena originale"); plt.axis('off')
plt.subplot(2,3,2); plt.imshow(lena_thresh_120, cmap='gray'); plt.title("Lena - seuil manuel 120"); plt.axis('off')
plt.subplot(2,3,3); plt.imshow(lena_otsu, cmap='gray'); plt.title("Lena - Otsu automatique"); plt.axis('off')

# Alex
plt.subplot(2,3,4); plt.imshow(alex, cmap='gray'); plt.title("Alex originale"); plt.axis('off')
plt.subplot(2,3,5); plt.imshow(alex_thresh_120, cmap='gray'); plt.title("Alex - seuil manuel 120"); plt.axis('off')
plt.subplot(2,3,6); plt.imshow(alex_otsu, cmap='gray'); plt.title("Alex - Otsu automatique"); plt.axis('off')

plt.tight_layout()
plt.show()

print("✅ Seuillage manuel et Otsu terminés avec succès!")
