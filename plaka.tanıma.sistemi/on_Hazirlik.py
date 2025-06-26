import cv2
import numpy as np
import math

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)#gauss filtresi icin 5 5 lik bir kernel filtresi uygulanacagını beliriyor
ADAPTIVE_THRESH_BLOCK_SIZE = 19# her pikselin eşik değerinin hesaplanacağı komşuluk alanının boyutudur.
ADAPTIVE_THRESH_WEIGHT = 9#Adaptif eşikleme işlemi sırasında, hesaplanan eşik değerinden çıkarılacak bir sabit değeri ifade eder.

#Kodun amacı, bir görüntüdeki karakterleri 
# tespit etmek ve tanımak için ön işleme yapmaktır. 
def onhazirlikislemi(orjinalresim):
    gri_ton_resim = extractValue(orjinalresim)#görüntüyü HSV (Hue, Saturation, Value) renk uzayına dönüştürür
    #ve bu uzaydan yalnızca Value (Parlaklık) bileşenini çıkarır.

    imgMaxContrastGrayscale = maximizeContrast(gri_ton_resim)# adım, gri tonlu görüntünün kontrastını artırmak için
    #Top-Hat ve Black-Hat morfolojik işlemlerini uygular.

    height, width = gri_ton_resim.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return gri_ton_resim, imgThresh
# end function

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function
#Renklerden bağımsız analiz yapmak,Parlaklık üzerinde işlem yapmak,
#Gürültü azaltma, kenar tespiti gibi işlemlere geçmek, gibi durumlarda parlaklık bileşeni kullanılır.
###################################################################################################
def maximizeContrast(imgGrayscale):
#kontrastı artırma: Görüntüdeki parlak ve karanlık bölgeleri daha belirgin hale getirir.
#Detayları vurgulama: Özellikle kenar tespiti, metin veya nesne çıkarımı gibi görevler için detayların öne çıkmasını sağlar.

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function