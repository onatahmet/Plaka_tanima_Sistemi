import cv2
import numpy as np
import math

###################################################################################################
class PossibleChar:
#Geometrik Özellikler: Konturun alanı, merkezi, köşegen uzunluğu ve en-boy oranı gibi bilgiler
#hesaplanır. Bu bilgiler, konturun bir karakter olup olmadığını anlamak için kullanılır.
    # constructor #################################################################################
    # Bir kontur (kapalı şekil) alır ve bunun geometrik özelliklerini hesaplar
    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(self.contour)
    # Konturu çevreleyen dikdörtgenin özelliklerini hesaplar (x, y, genişlik, yükseklik)
        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight
     # Dikdörtgenin alanı (genişlik × yükseklik)
        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight
    # Dikdörtgenin merkez noktası (X ve Y koordinatları)
        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2
    
        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))
    # Dikdörtgenin en-boy oranı (genişlik / yükseklik)
        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)
    # end constructor

# end class