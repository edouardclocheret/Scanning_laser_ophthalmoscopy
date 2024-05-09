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


def seuillage(img,img_mask,seuil):
    
    img_out = (img_mask & (img < seuil))
    return img_out


def seuillages_successifs (img, img_mask):
    #Cette fonction a pour but d'étudier quelle valeur de seuil est optimale
    
    for i in range(8):
        seuil = 40 +10*i
        img_out = seuillage(img,img_mask, seuil)

        alpha = 240+i+1
        plt.subplot(alpha)
        plt.imshow(img_out)
        title = 'Seuillage à '+ str(seuil)
        plt.title(title)
    plt.show()


def top_hat (img, elem_struct) :
    dilate = dilation(img)
    return img ^ dilate


def my_segmentation_test_boucles_pas_ouf(img, img_mask):
    print("seg")
    #Image Verite Terrain en booleen
    img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.bool_)
    
    img_out = img
    
    
    seuil = 70
    img_out = seuillage(img,img_mask,seuil)
    ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)

    
    
    print("Top_hat")
    for i in range(1,10,1):
        for j in range(1,10,1):
            elem = rectangle(i,j)
            

            img_out_test= white_tophat(img,elem)
            disk_ = disk(3)
            img_out = closing(img_out, disk_)



            ACCU_test, RECALL_test, img_out_skel, GT_skel = evaluate(img_out_test, img_GT)
            #reconnecter les vaisseaux
    
            if RECALL_test > RECALL:
                img_out = img_out +img_out_test
                ACCU = ACCU_test
                RECALL = RECALL_test
                print("amelioration")

            print(i,j)
    
    
            
    
    

            
            
    
    
    return img_out
    
def my_segmentation(img, img_mask):

    img_out = img
    
    
    
    #objectif : mettre en valeur les structures linéaires
    elem = rectangle(1,50)
    elem_2 = rectangle(50,1)
    elem_3 = np.eye(50, dtype=bool)
    img_out= black_tophat(img_out,elem)+black_tophat(img_out,elem_2)    
    
    #disk_elem = disk(2)
    #img_out = closing(img_out, disk_elem)
  
    #objectif : supprimer les points isolés
    img_out = opening(img_out, disk(2))

    seuil = 20
    img_out = seuillage(img_out,img_mask,seuil)

    
    
    
    #applique le masque
    seuil = 1
    img_out = seuillage(img_out,img_mask,seuil)
    
    plt.imshow(img_out)
    plt.show()
    
    return img_out



def evaluate(img_out, img_GT):
    #marche seulement ave le changement de type 
    GT_skel = skeletonize(1.0 * img_GT) # On reduit le support de l'evaluation...
    img_out_skel = skeletonize(img_out) # ...aux pixels des squelettes
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs
    
    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel


def affichage_my_seg(img,img_out,img_out_skel,img_GT,GT_skel):
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


def main() : 
    #Ouvrir l'image originale en niveau de gris
    img =  np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    
    #On ne considere que les pixels dans le disque inscrit 
    img_mask = (np.ones(img.shape)).astype(np.bool_)
    invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
    img_mask[invalid_pixels] = 0

    img_out = my_segmentation(img,img_mask)

    #Ouvrir l'image Verite Terrain en booleen
    img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.bool_)
    
    ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
    

    #affichage_my_seg(img,img_out,img_out_skel,img_GT,GT_skel)
    print("Accuracy = ", ACCU, "Recall =", RECALL)
    


if __name__ == "__main__" :
    main()