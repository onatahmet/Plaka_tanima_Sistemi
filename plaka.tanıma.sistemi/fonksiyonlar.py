import cv2
import numpy as np



def resimAc(sec):
    # Kullanıcıdan dosya adını tam olarak alıyoruz
    img_path = "Resim/" + sec  # Kullanıcının tam dosya adını (uzantı dahil) kullanıyoruz
    
    # Dosyadan resmi okuma
    img = cv2.imread(img_path)
    if img is None:  # Resim yüklenemezse hata mesajı ver
        print(f"Hata: '{img_path}' yolu bulunamadi veya desteklenmeyen bir format!")
        return None
    
    cv2.namedWindow("1-Orjinal Resim", cv2.WINDOW_NORMAL)
    # Resmi gösterme
    cv2.imshow("1-Orjinal Resim", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

# RGB uzayından Gri seviyeli resme dönüş işlemi
def griyecevir(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Griye dönüştürme işlemi
    cv2.namedWindow("2-Griye Donusturme Islemi", cv2.WINDOW_NORMAL)
    # Pencre Oluştur
    cv2.imshow("2-Griye Donusturme Islemi", img_gray)
    # Resmi Göster
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_gray

# Gürültü azaltıcı yumuşatma işlemi
def gurultuAzalt(img_gray):
    gurultuazalt = cv2.bilateralFilter(img_gray, 9, 75, 75)
    cv2.namedWindow("3-Gurultu Temizleme Islemi", cv2.WINDOW_NORMAL)
    cv2.imshow("3-Gurultu Temizleme Islemi", gurultuazalt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gurultuazalt

# Histogram eşitleme işlemi
def histogramEsitleme(gurultuazalt):
    histogram_e = cv2.equalizeHist(gurultuazalt)
    cv2.namedWindow("4-Histogram esitleme islemi", cv2.WINDOW_NORMAL)
    cv2.imshow("4-Histogram esitleme islemi", histogram_e)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return histogram_e

# Morfolojik açma işlemi
def morfolojikIslem(h_esitleme):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morfolojikresim = cv2.morphologyEx(h_esitleme, cv2.MORPH_OPEN, kernel, iterations=15)
    cv2.namedWindow("5-Morfolojik acilim", cv2.WINDOW_NORMAL)
    cv2.imshow("5-Morfolojik acilim", morfolojikresim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return morfolojikresim

# Görüntü çıkarma işlemi
def goruntuCikarma(h_esitleme, morfolojik_resim):
    gcikarilmisresim = cv2.subtract(h_esitleme, morfolojik_resim)
    cv2.namedWindow("6-Goruntu cikarma", cv2.WINDOW_NORMAL)
    cv2.imshow("6-Goruntu cikarma", gcikarilmisresim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gcikarilmisresim

# Görüntüyü eşik değerine göre siyah-beyaz olarak ayıran işlem
def goruntuEsikle(goruntucikarma):
    ret, goruntuesikle = cv2.threshold(goruntucikarma, 0, 255, cv2.THRESH_OTSU)
    cv2.namedWindow("7-Goruntu Esikleme", cv2.WINDOW_NORMAL)
    cv2.imshow("7-Goruntu Esikleme", goruntuesikle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return goruntuesikle

# Canny kenar tespiti işlemi
def cannyEdge(goruntuesikleme):
    canny_goruntu = cv2.Canny(goruntuesikleme, 250, 255)
    cv2.namedWindow("8-Canny Edge", cv2.WINDOW_NORMAL)
    cv2.imshow("8-Canny Edge", canny_goruntu)
    canny_goruntu = cv2.convertScaleAbs(canny_goruntu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return canny_goruntu

# Dilatasyon işlemi
def genisletmeIslemi(cannedge_goruntu):
    cekirdek = np.ones((3, 3), np.uint8)
    gen_goruntu = cv2.dilate(cannedge_goruntu, cekirdek, iterations=1)
    cv2.namedWindow("9-Genisletme", cv2.WINDOW_NORMAL)
    cv2.imshow("9-Genisletme", gen_goruntu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gen_goruntu

def konturIslemi(img, gen_goruntu):
    try:
        contours_data = cv2.findContours(gen_goruntu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[-2]
        hierarchy = contours_data[-1]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)

            if len(approx) == 4:  # 4 köşe tespiti
                screenCnt = approx
                break

        if screenCnt is not None:
            final = cv2.drawContours(img.copy(), [screenCnt], -1, (9, 236, 255), 3)
            cv2.namedWindow("10-Konturlu Goruntu", cv2.WINDOW_NORMAL)
            cv2.imshow("10-Konturlu Goruntu", final)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Plaka konturu bulunamadi.")

        return screenCnt

    except Exception as e:
        print(f"Kontur isleminde bir hata olustu: {e}")
        return None

def maskelemeIslemi(img_gray, img, screenCnt):
    if screenCnt is None or not isinstance(screenCnt, np.ndarray) or len(screenCnt) == 0:
        print("Kontur bulunamadi, maskeleme yapilamaz.")
        return None

    # Maske oluştur ve sadece plaka alanını tut
    mask = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    yeni_goruntu = cv2.bitwise_and(img, img, mask=mask)

    # Maskeleme sonucunu göster
    cv2.namedWindow("11-Plaka", cv2.WINDOW_NORMAL)
    cv2.imshow("11-Plaka", yeni_goruntu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return yeni_goruntu

def plakaIyilestir(yeni_goruntu):
    if yeni_goruntu is None:
        print("Plaka görüntüsü mevcut değil, iyileştirme yapılamaz.")
        return None

    # YCrCb renk uzayına dönüştür ve kanalları ayır
    y, cr, cb = cv2.split(cv2.cvtColor(yeni_goruntu, cv2.COLOR_BGR2YCrCb))

    # Y kanalında histogram eşitleme
    y = cv2.equalizeHist(y)

    # Kanalları birleştir ve geri RGB'ye dönüştür
    son_resim = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # İyileştirilmiş görüntüyü göster
    cv2.namedWindow("Gelismis Plaka", cv2.WINDOW_NORMAL)
    cv2.imshow("Gelismis Plaka", son_resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return son_resim
