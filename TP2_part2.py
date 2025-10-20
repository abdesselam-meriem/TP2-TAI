# ==============================================================
# TP2 - Partie 2 : Histogrammes et transformations d’intensité
# ==============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger les images en niveaux de gris
lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
alex = cv2.imread('alex.png', cv2.IMREAD_GRAYSCALE)
if lena is None or alex is None:
    raise FileNotFoundError("Assure-toi que lena.jpg et alex.png sont dans le dossier.")

# 2. Calcul manuel de l’histogramme
def HISTO(img):
    hist = np.zeros(256, dtype=int)
    for p in img.flatten():
        hist[p] += 1
    return hist

# 3. Affichage image + histogramme
def show_image_and_hist(img, title):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.bar(range(256), HISTO(img), width=1)
    plt.title("Histogramme")
    plt.tight_layout()
    plt.show()

# Histogramme original
show_image_and_hist(lena, "Lena (originale)")

# 4. Transformation TRL : ajout d’une constante
def TRL(img, C):
    r = img.astype(np.int16) + C
    r = np.clip(r, 0, 255)
    return r.astype(np.uint8)

lena_plus50 = TRL(lena, 50)
lena_minus50 = TRL(lena, -50)

show_image_and_hist(lena_plus50, "Lena +50")
show_image_and_hist(lena_minus50, "Lena -50")

# 5. Inversion
lena_inv = 255 - lena
show_image_and_hist(lena_inv, "Lena inversée")

# ==============================================================
# Étirement de contraste (alex.png)
# ==============================================================

def contrast_stretch(img):
    Imin, Imax = np.min(img), np.max(img)
    stretched = (img - Imin) * 255.0 / (Imax - Imin)
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)

alex_stretched = contrast_stretch(alex)
show_image_and_hist(alex, "Alex (originale)")
show_image_and_hist(alex_stretched, "Alex étirée")

# ==============================================================
# Seuillage manuel et automatique (Otsu)
# ==============================================================

def threshold_manual(img, t):
    res = np.zeros_like(img)
    res[img >= t] = 255
    return res

t_lena = 100
t_alex = 120
lena_thresh = threshold_manual(lena, t_lena)
alex_thresh = threshold_manual(alex, t_alex)

show_image_and_hist(lena_thresh, f"Lena seuillage manuel (t={t_lena})")
show_image_and_hist(alex_thresh, f"Alex seuillage manuel (t={t_alex})")

# Otsu
_, lena_otsu = cv2.threshold(lena, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, alex_otsu = cv2.threshold(alex, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

show_image_and_hist(lena_otsu, "Lena (Otsu)")
show_image_and_hist(alex_otsu, "Alex (Otsu)")

# Comparaison manuelle vs Otsu
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1); plt.imshow(lena_thresh, cmap='gray'); plt.title(f"Lena - manuel (t={t_lena})"); plt.axis('off')
plt.subplot(2, 2, 2); plt.imshow(lena_otsu, cmap='gray'); plt.title("Lena - Otsu"); plt.axis('off')
plt.subplot(2, 2, 3); plt.imshow(alex_thresh, cmap='gray'); plt.title(f"Alex - manuel (t={t_alex})"); plt.axis('off')
plt.subplot(2, 2, 4); plt.imshow(alex_otsu, cmap='gray'); plt.title("Alex - Otsu"); plt.axis('off')
plt.tight_layout()
plt.show()
