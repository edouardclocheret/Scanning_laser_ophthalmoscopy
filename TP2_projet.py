import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt



def my_segmentation(img, img_mask, seuil):
    img_out = (img_mask & (img < seuil))
    return img_in

def evaluate(img_out, img_GT):
    print("evaluate")
    #marche seulement ave le changement de type 
    GT_skel = skeletonize(1.0 * img_GT) # On reduit le support de l'evaluation...
    print("2")
    img_out_skel = skeletonize(img_out) # ...aux pixels des squelettes
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs
    
    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

#Ouvrir l'image originale en niveau de gris
img =  np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask,80)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.bool_)
print(img_GT.shape)
ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU,', Recall =', RECALL)

def affichage(img,img_out,img_out_skel,img_GT,GT_skel):
    plt.subplot(231)
    plt.imshow(img,cmap = 'gray')
    plt.title('Image Originale')
    plt.subplot(232)
    plt.imshow(img_out)
    plt.title('Segmentation')
    plt.subplot(233)
    plt.imshow(img_out_skel)
    plt.title('Segmentation squelette')
    plt.subplot(235)
    plt.imshow(img_GT)
    plt.title('Verite Terrain')
    plt.subplot(236)
    plt.imshow(GT_skel)
    plt.title('Verite Terrain Squelette')
    plt.show()
