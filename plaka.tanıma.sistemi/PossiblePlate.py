import cv2
import numpy as np

###################################################################################################
#Yapıcı (constructor) metodudur.
# Yeni bir PossiblePlate nesnesi oluşturulduğunda, bu metod otomatik olarak çalıştırılır
#ve sınıfın özelliklerini (attributes) başlangıç değerleriyle başlatır.
class PossiblePlate:

    # constructor #################################################################################
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None
# Plaka üzerindeki karakterler, bir metin (string) olarak depolanır
# Örneğin, "34ABC123" gibi.
        self.strChars = ""
    # end constructor

# end class