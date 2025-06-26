# PlakalariTespitEt.py
import os
import cv2
import numpy as np
import math
import Main
import random

import on_Hazirlik
import KarakterTespitEt
import PossiblePlate
import PossibleChar
import mainK

# Plaka genişliği ve yüksekliği için ekleme faktörleri (padding
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

###################################################################################################

print("Çalisma dizini:", os.getcwd())

def plaka_tespit_et(imgOriginalScene):
    listOfPossiblePlates = []                   # # Bulunan olası plakaların listesi

    height, width, numChannels = imgOriginalScene.shape
    # Giriş görüntüsünün boyutlarını al ve boş matrisler oluştur
    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.adimleri_goster == True: # Eğer adımları göstermek etkinse, orijinal sahneyi göster
        cv2.imshow("0", imgOriginalScene)
    # end if # 
      # Gri ton ve eşiklenmiş görüntüyü oluştur (ön işleme)
    imgGrayscaleScene, imgThreshScene = on_Hazirlik.onhazirlikislemi(imgOriginalScene)         

    if Main.adimleri_goster == True: # show steps #######################################################
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    # end if # show steps #########################################################################

    # Görüntüdeki olası karakterleri bul
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if Main.adimleri_goster == True: # show steps #######################################################
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))  # 

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []
    # Olası karakterlerin konturlarını çiz
        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, mainK.beyaz)
        cv2.imshow("2b", imgContours)
    # end if # show steps #########################################################################

     # Eşleşen karakter gruplarını bul     
    listOfListsOfMatchingCharsInScene = KarakterTespitEt.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.adimleri_goster == True: # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)
     # Her grup için rastgele bir renk seç ve konturları çiz
        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if # show steps #########################################################################
    # Eşleşen her karakter grubu için plaka çıkarmaya çalış
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)    # Plaka çıkarımı     

        if possiblePlate.imgPlate is not None:                        
            listOfPossiblePlates.append(possiblePlate)                  
        # end if
    # end for

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  # 13 with MCLRNF1 image

    if Main.adimleri_goster == True: # show steps #######################################################
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), mainK.kirmizi, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), mainK.kirmizi, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), mainK.kirmizi, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), mainK.kirmizi, 2)

            cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print("\nPlaka tespiti tamamlandi, herhangi bir resme tikla ve bir tuşa tikla ve karakter tanimayi başlat . . .\n")
        cv2.waitKey(0)
    # end if # show steps #########################################################################

    return listOfPossiblePlates
# end function

###################################################################################################
def findPossibleCharsInScene(imgThresh):
    # Bu fonksiyon, verilen threshold uygulanmış bir görüntüde olası karakter konturlarını bulur.
    listOfPossibleChars = []   # Olası karakterlerin tutulacağı liste             

    intCountOfPossibleChars = 0 # Bulunan olası karakterlerin sayısı

    imgThreshCopy = imgThresh.copy()# Orijinal threshold görüntüsünün bir kopyası oluşturulur

    # Tüm konturlar bulunur
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Görüntü boyutları alınır ve aynı boyutlarda boş bir görüntü oluşturulur
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
     # Her bir kontur üzerinde işlem yapılır
    for i in range(0, len(contours)):                    

        if Main.adimleri_goster == True: 
            cv2.drawContours(imgContours, contours, i, mainK.beyaz) # Kontur çizilir
        # end if
        # Kontur, PossibleChar sınıfına dönüştürülür
        possibleChar = PossibleChar.PossibleChar(contours[i])
         # Eğer kontur bir olası karakter olarak uygun ise
        if KarakterTespitEt.checkIfPossibleChar(possibleChar):   # Olası karakter sayısı artırılır               
            intCountOfPossibleChars = intCountOfPossibleChars + 1           
            listOfPossibleChars.append(possibleChar) # Listeye eklenir                      
        # end if
    # end for

    if Main.adimleri_goster == True: # show steps #######################################################
        print("\nstep 2 - len(contours) = " + str(len(contours)))  # Bulunan toplam kontur sayısı
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))# Olası karakter sayısı
        cv2.imshow("2a", imgContours) # Konturların görüntüsü gösterilir
    # end if # show steps #########################################################################

    return listOfPossibleChars
# end function


###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
     # Bu fonksiyon, eşleşen karakterlerden oluşan bir gruptan plaka bölgesini çıkarır.
    possiblePlate = PossiblePlate.PossiblePlate()  # Dönüş değeri olarak kullanılacak PossiblePlate nesnesi
 # Eşleşen karakterler x koordinatlarına göre soldan sağa sıralanır
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)  
# Plakanın merkez noktası hesaplanır      
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

# Plakanın genişlik ve yüksekliği hesaplanır
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0
     # Tüm karakterlerin yüksekliği toplanır
    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight# Ortalama karakter yüksekliği hesaplanır
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)# Plaka yüksekliği hesaplanır

    # Plaka bölgesinin düzeltme açısı hesaplanır
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = KarakterTespitEt.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # Plaka bölgesi bilgileri PossiblePlate nesnesine atanır       
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # Plaka bölgesinin rotasyonu gerçekleştirilir       
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
    # Orijinal görüntünün boyutları alınır
    height, width, numChannels = imgOriginal.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
     # Kırpılan plaka görüntüsü PossiblePlate nesnesine atanır
    possiblePlate.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate

    return possiblePlate# PossiblePlate nesnesi döndürülür
# end function