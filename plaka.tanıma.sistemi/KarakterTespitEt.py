# KarakterTespitEt.py
import os

import cv2
import numpy as np
import math
import random
import Main
import on_Hazirlik
import PossibleChar
import mainK
import veri_tabani
from datetime import datetime


# cv2'nin K-Nearest modelini olusturuyoruz, bu model karakter tanimlama icin kullanilacak
kNearest = cv2.ml.KNearest_create()

# Belirli bir konturun "muhtemel bir karakter" olup olmadigini kontrol etmek icin kullanilan sabitler        
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

      
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

     
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

###################################################################################################
def KNN_verisi_yukle_KNN_ogren():
    allContoursWithData = []                # Tüm kontur verileri için boş bir liste oluştur.
    validContoursWithData = []              # Geçerli kontur verileri için boş bir liste oluştur.
    # Sınıflandırma verilerini yükle (karakter sınıflarını içeren dosya)
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                   # Eğitim sınıflandırmalarını oku
    except:                                                                               
        print("error, unable to open classifications.txt, exiting program\n")  
        os.system("pause")
        return False                                                                        
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)    # Eğitim görüntülerini oku            
    except:                                                                                
        print("error, unable to open flattened_images.txt, exiting program\n")  
        os.system("pause")
        return False                                                                        
    # end try
     # Sınıflandırma verilerini, KNN'in eğitim fonksiyonuna uygun formata dönüştür
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       

    kNearest.setDefaultK(1)        # KNN için varsayılan K değerini 1 olarak ayarla                                                    

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)     # KNN objesini eğit

    return True            
# end function

if not os.path.exists("classifications.txt"):
    print("classifications.txt bulunamadi!")
if not os.path.exists("flattened_images.txt"):
    print("flattened_images.txt bulunamadi!")

    
###################################################################################################
def plakada_karakter_tespit_et(listOfPossiblePlates):
    # Her bir olası plaka üzerinde karakter tespiti yapılacak
    intPlateCounter = 0
    imgContours = None
    contours = []
     # Eğer olası plakaların listesi boşsa, hiçbir işlem yapmadan geri dön
    if len(listOfPossiblePlates) == 0:          
        return listOfPossiblePlates             
    # end if

            

    for possiblePlate in listOfPossiblePlates:          
         # Plaka bölgesini ön işleme tabi tutarak gri tonlamalı ve eşiklenmiş görüntüler elde edilir
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = on_Hazirlik.onhazirlikislemi(possiblePlate.imgPlate)    

        if Main.adimleri_goster == True: # show steps ###################################################
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if # show steps #####################################################################

        # Plaka görüntüsünü büyüt (karakterlerin daha kolay algılanabilmesi için)
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

         # Gri alanları tamamen yok etmek için yeniden eşikleme uygula       
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.adimleri_goster == True: 
            cv2.imshow("5d", possiblePlate.imgThresh)
        # end if 

        # Plaka içinde olası karakterleri bul
        # Bu işlem, önce tüm konturları bulur ve daha sonra karakter olabilecek konturları filtreler        
                
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if Main.adimleri_goster == True: 
             # Olası plakanın boyut bilgilerini al
            height, width, numChannels = possiblePlate.imgPlate.shape# Konturları göstermek için boş bir görüntü oluştur
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]# Daha önceki konturları temizle

             # Plaka içindeki olası karakter konturlarını konturlar listesine ekle
            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for
            # Olası karakterlerin konturlarını çizecek
            cv2.drawContours(imgContours, contours, -1, mainK.beyaz)

            cv2.imshow("6", imgContours)
        # end if # show steps #####################################################################

         # Plakadaki tüm olası karakterlere göre eşleşen karakter gruplarını bul       
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if Main.adimleri_goster == True: 
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]
            # Eşleşen karakter gruplarını rastgele renklerle görselleştir
            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if 

        if (len(listOfListsOfMatchingCharsInPlate) == 0):# Eğer plaka içinde eşleşen karakter grupları bulunamazsa

            if Main.adimleri_goster == True:
                print("chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1 # Plaka sayacını artır
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            

            possiblePlate.strChars = ""# Eşleşen karakter olmadığı için karakter dizisi boş bırakılır
            continue						
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):  # Her bir eşleşen karakter grubunda
            # Karakterleri soldan sağa sıralar                            
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)    
            # İç içe geçmiş ve çakışan karakterleri çıkarır    
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])             
        # end for

        if Main.adimleri_goster == True: # Yeni bir boş görüntü oluştur
            imgContours = np.zeros((height, width, 3), np.uint8)
            # Rastgele bir renk seçilir

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]# Konturları temizle

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if 

                
        intLenOfLongestListOfChars = 0# En uzun karakter dizisinin uzunluğunu tutar
        intIndexOfLongestListOfChars = 0 # En uzun karakter dizisinin indeksini tutar

        # En uzun karakter dizisini bulmak için tüm listeleri dolaşır
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        # Plakadaki en uzun ve en anlamlı karakter dizisini seç        
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main.adimleri_goster == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, mainK.beyaz)

            cv2.imshow("9", imgContours)
        # end if # show steps #####################################################################
        #seçilen karakter grubunu OCR (Optical Character Recognition) algoritması kullanarak tanır ve karakterleri bir string olarak kaydeder.
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)
        if Main.adimleri_goster == True: # Adımları görselleştir 
            print("chars found in plate number " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if # show steps #####################################################################

    
    #karakter tespit sürecinin tamamlandığını kullanıcıya bildirir ve plaka adaylarının listesini döndürür.

    if Main.adimleri_goster == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function


def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []        # Dönüş değeri olacak, olası karakterleri içeren liste
    contours = []       # Konturları saklamak için bir liste
    imgThreshCopy = imgThresh.copy()# Eşiklenmiş görüntünün bir kopyasını oluştur

    # Plaka içerisindeki tüm konturları bul       
    
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Her bir kontur için
    for contour in contours:                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)
        # Eğer kontur bir karakter olabilir ise, listeye ekle
        if checkIfPossibleChar(possibleChar):             
            listOfPossibleChars.append(possibleChar)       
        # end if
    # end if

    return listOfPossibleChars
# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
            
          
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and# Minimum alan kontrolü
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and# Minimum genişlik kontrolü
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):# Minimum yükseklik kontrolü
        return True
    else:
        return False
    # end if
# end function

###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):
            
            
           
    listOfListsOfMatchingChars = []                  

    for possibleChar in listOfPossibleChars:    
         # Mevcut karakterle eşleşen karakterleri bul                   
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        
           # Mevcut karakteri de eşleşen listeye ekle
        listOfMatchingChars.append(possibleChar)               
        # Eğer eşleşen karakterlerin sayısı bir plaka için yeterince fazla değilse
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue                           
                                               
        # end if

        # Eğer buraya gelindiyse, geçerli liste bir grup olarak kabul edilir                                      
        listOfListsOfMatchingChars.append(listOfMatchingChars)      
         # Listeyi ana listeye ekle
        listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                
         # Eşleşen karakterleri büyük listeden çıkar                                       
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        # Geri kalan karakterlerle yeniden aynı işlemi uygula (rekürsif çağrı)
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call
        # Rekürsif çağrı ile bulunan listeleri ana listeye ekle
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars
        # end for

        break       # exit for

    # end for
    # Eşleşen karakter gruplarını döndür
    return listOfListsOfMatchingChars
# end function
# Bu fonksiyon, tek bir olası karakteri ve büyük bir karakter listesini alarak,
# olası karaktere eşleşen tüm karakterleri bulur ve bu karakterleri bir liste olarak döndürür.
def findListOfMatchingChars(possibleChar, listOfChars):
         

    listOfMatchingChars = []            # Eşleşen karakterlerin tutulacağı liste


    for possibleMatchingChar in listOfChars:        # Büyük liste üzerindeki her karakteri kontrol et
        if possibleMatchingChar == possibleChar:    # Eğer büyük listedeki karakter, aradığımız karakter ile aynıysa
                                                    
            continue                    # Bu durumda, kendisini eşleşen karakterler listesine eklemeyip, döngünün başına geri dön            
        # end if
        # Şimdi iki karakter arasında mesafe, açı ve boyut değişikliklerini hesaplayacağız            
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        # Karakterlerin eşleşip eşleşmediğini kontrol et
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        
        # end if
    # end for

    return listOfMatchingChars                 # Sonuç olarak eşleşen karakterleri döndür
# end function

###################################################################################################
# Kullanıcılar arasındaki mesafeyi hesaplamak için Pisagor teoremi kullanılır
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# Temel trigonometri (SOH CAH TOA) kullanarak karakterler arasındaki açıyı hesapla
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                          
        fltAngleInRad = math.atan(fltOpp / fltAdj)      
    else:
        fltAngleInRad = 1.5708                          
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)      

    return fltAngleInDeg    # Açıyı derece cinsinden döndür
# end function

###################################################################################################

# Eğer iki karakter örtüşüyorsa veya çok yakınsa, içteki (daha küçük) karakteri kaldır
# Bu, aynı karakterin birden fazla sayılmasını önler (örneğin, 'O' harfi iç halka ve dış halka ile iki kez bulunabilir)
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)  # Sonuç listesini başlat        

    for currentChar in listOfMatchingChars:# Her bir karakteri kontrol et
        for otherChar in listOfMatchingChars:# Diğer karakterleri de kontrol et
            if currentChar != otherChar:    # Eğer şu anki karakter diğerinden farklıysa    
                                                                            
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                               # Eğer iki karakter çok yakınsa, hangisinin daha küçük olduğunu kontrol et
                               
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # Eğer şu anki karakter daha küçükse
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # Eğer daha önce çıkarılmamışsa
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)        # Şu anki karakteri kaldır
                        # end if
                    else:                                                                      
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:          # Eğer daha önce çıkarılmamışsa      
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)       # Şu anki karakteri kaldır    
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function

###################################################################################################
# Bu fonksiyon, plakanın üzerindeki karakterleri tanımak için kullanılır.
# Verilen eşleşen karakterler listesindeki her bir karakterin görüntüdeki yerini analiz eder ve tanıdık karakteri döndürür.
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""          
     

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     

    for currentChar in listOfMatchingChars:                                         
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, mainK.yesil, 2)           

                # crop char out of threshold image
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        

        npaROIResized = np.float32(npaROIResized)               

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              

        strCurrentChar = str(chr(int(npaResults[0][0])))            

        strChars = strChars + strCurrentChar                       

    # end for

    if Main.adimleri_goster == True: 
        cv2.imshow("10", imgThreshColor)
    # end if 

    return strChars
# end function

