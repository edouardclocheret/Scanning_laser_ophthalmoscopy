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

def my_segmentation(img, img_mask):

    img_out = img
    
    
    #Objectif : mettre en valeur les structures linéaires
    #On utilise donc de longs rectangles comme éléments structurants
    elem = rectangle(1,50)
    elem_2 = rectangle(50,1)

    #On travaille à différentes échelles pour couvrir une plus large gamme de vaisseaux
    elem_3 = rectangle(1,5)
    elem_4 = rectangle(5,1)

    #Prend aussi en compte les elements orientés à 45 degrés
    elem_5 = np.eye(50, dtype=bool)

    #Le top hat met en évidence les éléments fins d'une image 
    #et réduit l'importance des zones homogènes
    img_out= black_tophat(img_out,elem)+black_tophat(img_out,elem_2) + \
        black_tophat(img_out,elem_3)+black_tophat(img_out,elem_4)  + \
        +black_tophat(img_out,elem_5)
    
    #Conversion de l' image en niveaux de gris en binaire 
    seuil = 70
    img_out = seuillage(img_out,img_mask,seuil)

    #Objectif : supprimer les points isolés
    img_out = closing(img_out,disk(1))
    img_out = opening(img_out, disk(3))
    
    #Applique le masque et prend le négatif
    seuil = 1
    img_out = seuillage(img_out,img_mask,seuil)
    
    plt.imshow(img_out)
    plt.show()
    
    return img_out


def my_segmentation_optimisation(img, img_mask):
    
    img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.bool_)
    
    img_out = img
    elem = rectangle(1,50)
    elem_2 = rectangle(50,1)
    elem_3 = rectangle(1,5)
    elem_4 = rectangle(5,1)
    elem_5 = np.eye(50, dtype=bool)

    img_out= black_tophat(img_out,elem)+black_tophat(img_out,elem_2) + \
        black_tophat(img_out,elem_3)+black_tophat(img_out,elem_4)  + \
        +black_tophat(img_out,elem_5)

    seuil = 70
    img_out = seuillage(img_out,img_mask,seuil)

    img_out = closing(img_out,disk(1))
    img_out = opening(img_out, disk(3))
    seuil = 1
    img_out = seuillage(img_out,img_mask,seuil)
    img
    ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
    seuil_param_best =70
    fermeture_param_best=1
    ouverture_param_best =3

    for seuil_param in range(60,85,5):
        for fermeture_param in range(1,6,1):
            for ouverture_param in range(1,6,1):
                print(seuil_param, fermeture_param, ouverture_param)
                img_out_test = img
                elem = rectangle(1,50)
                elem_2 = rectangle(50,1)
                elem_3 = rectangle(1,5)
                elem_4 = rectangle(5,1)
                elem_5 = np.eye(50, dtype=bool)

                img_out_test= black_tophat(img_out_test,elem)+black_tophat(img_out_test,elem_2) + \
                    black_tophat(img_out_test,elem_3)+black_tophat(img_out_test,elem_4)  + \
                    +black_tophat(img_out_test,elem_5)

                seuil = seuil_param
                img_out_test = seuillage(img_out_test,img_mask,seuil)

                img_out_test = closing(img_out_test,disk(fermeture_param))
                img_out_test = opening(img_out_test, disk(ouverture_param))
                
                #pour prendre le négatif
                seuil = 1
                img_out_test = seuillage(img_out_test,img_mask,seuil)
                
                ACCU_test, RECALL_test, img_out_skel, GT_skel = evaluate(img_out, img_GT)
    
    
                if RECALL_test+ACCU_test > RECALL+ACCU:
                    img_out = img_out_test
                    ACCU = ACCU_test
                    RECALL = RECALL_test
                    seuil_param_best =seuil_param
                    fermeture_param_best=fermeture_param
                    ouverture_param_best = ouverture_param

                    print("amelioration")    
    
    print("Meilleurs paramètres :")
    print("Seuil", seuil_param_best)
    print("Rayon Fermeture", fermeture_param_best)
    print("Rayon ouverture", ouverture_param_best)
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

    #Pour la recherche des paramètres optimaux :
    #img_out = my_segmentation_optimisation(img,img_mask)


    #Accuracy et recall
    img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.bool_)
    ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
    
    #Afichage du squelette, vérité terrain etc
    affichage_my_seg(img,img_out,img_out_skel,img_GT,GT_skel)
    print("Accuracy = ", ACCU, "Recall =", RECALL)
    


if __name__ == "__main__" :
    main()