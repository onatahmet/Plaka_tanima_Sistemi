import cv2
import numpy as np
import os
import KarakterTespitEt
import PlakalariTespitEt
import PossiblePlate
import datetime
import veri_tabani
from flask import Flask, request, jsonify
from flask_cors import CORS

# module level variables
siyah = (0.0, 0.0, 0.0)
beyaz = (255.0, 255.0, 255.0)
sari = (0.0, 255.0, 255.0)
yesil = (0.0, 255.0, 0.0)
kirmizi = (0.0, 0.0, 255.0)

print("Çalışma dizini:", os.getcwd())

app = Flask(__name__)
CORS(app)  # Farklı portlardan gelen istekleri kabul et

# ---------- API ENDPOINTLERİ ----------

@app.route('/')
def index():
    return "Plaka Tanıma Sistemi API'ye hoş geldiniz!"

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/api/recognize-plate', methods=['POST'])
def recognize_plate():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    ihtimal_plaka_listeleri = PlakalariTespitEt.plaka_tespit_et(image)
    ihtimal_plaka_listeleri = KarakterTespitEt.plakada_karakter_tespit_et(ihtimal_plaka_listeleri)

    if len(ihtimal_plaka_listeleri) == 0:
        return jsonify({"status": "error", "message": "Plaka tespit edilemedi"}), 400

    ihtimal_plaka_listeleri.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
    plaka = ihtimal_plaka_listeleri[0].strChars

    tarih_saat = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    veri_tabani.plaka_ekle(plaka, tarih_saat)

    return jsonify({"status": "success", "plate": plaka})

@app.route('/api/get-plates', methods=['GET'])
def get_plates():
    # Veritabanından plakaları listele
    plakalar = veri_tabani.plakalari_listele()
    
    # Plakalardan JSON objesi oluştur
 
    return jsonify({"status": "success", "plates": plakalar}), 200

# ---------- GÖRSEL ARAYÜZ KULLANIMI (İSTERSEN) ----------
def main():
    KNN_Ogrenme_basarisi = KarakterTespitEt.KNN_verisi_yukle_KNN_ogren()
    if KNN_Ogrenme_basarisi == False:
        print("\nhata: KNN başarılı uygulanamadı\n")
        return

    orjinal_resim = cv2.imread("Resim/d.png")
    if orjinal_resim is None:
        print("\nhata: dosyadan resim okunamadı \n\n")
        os.system("pause")
        return

    ihtimal_plaka_listeleri = PlakalariTespitEt.plaka_tespit_et(orjinal_resim)
    ihtimal_plaka_listeleri = KarakterTespitEt.plakada_karakter_tespit_et(ihtimal_plaka_listeleri)

    cv2.imshow("orjinal_resim", orjinal_resim)

    if len(ihtimal_plaka_listeleri) == 0:
        print("\nPlaka tespit edilemedi\n")
    else:
        ihtimal_plaka_listeleri.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        Plaka = ihtimal_plaka_listeleri[0]
        cv2.imshow("imgPlate", Plaka.imgPlate)
        cv2.imshow("imgThresh", Plaka.imgThresh)

        if len(Plaka.strChars) == 0:
            print("\nkarakter tespit edilemedi.\n\n")
            return

        PlakaCevresineKirmiziDortgenCiz(orjinal_resim, Plaka)
        print("\nresimden okunan  plaka = " + Plaka.strChars + "\n")
        print("----------------------------------------")
        resimePlakalariIsle(orjinal_resim, Plaka)
        cv2.imshow("orjinal_resim", orjinal_resim)
        cv2.imwrite("orjinal_resim.png", orjinal_resim)

    cv2.waitKey(0)

# ---------- Yardımcı Fonksiyonlar ----------
def PlakaCevresineKirmiziDortgenCiz(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    p2fRectPoints = p2fRectPoints.astype(np.int32)
    for i in range(4):
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[i]), tuple(p2fRectPoints[(i+1)%4]), kirmizi, 2)

def resimePlakalariIsle(imgOriginalScene, licPlate):
    sceneHeight, sceneWidth, _ = imgOriginalScene.shape
    plateHeight, plateWidth, _ = licPlate.imgPlate.shape
    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    ((intPlateCenterX, intPlateCenterY), _, _) = licPlate.rrLocationOfPlateInScene
    intPlateCenterX, intPlateCenterY = int(intPlateCenterX), int(intPlateCenterY)

    ptCenterOfTextAreaX = intPlateCenterX
    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    cv2.putText(imgOriginalScene, licPlate.strChars,
                (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY),
                intFontFace, fltFontScale, sari, intFontThickness)

    plaka_metni = licPlate.strChars
    tarih_saat = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    veri_tabani.plaka_ekle(plaka_metni, tarih_saat)

# ---------- Çalıştırma ----------
if __name__ == "__main__":
    main()  # Görsel işleme fonksiyonunu çalıştır  
    app.run(debug=True)  # Flask sunucusunu başlat 