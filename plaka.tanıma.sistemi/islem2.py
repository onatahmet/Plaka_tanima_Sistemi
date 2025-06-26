import fonksiyonlar as fonk

def goruntu():
    try:
        sec = input("Resim Adi:")
        if sec:  # Seçilen dosya adı boş değilse devam et
            try:
                img = fonk.resimAc(sec)  # Resmi açmaya çalış
            except Exception as e:
                print(f"Resim açilirken bir hata oluştu: {e}")
                return
            
            try:
                img_gray = fonk.griyecevir(img)  # Griye çevirme işlemi
            except Exception as e:
                print(f"Griye çevirme işleminde bir hata oluştu: {e}")
                return
            
            try:
                gurultuazalt = fonk.gurultuAzalt(img_gray)  # Gürültü azaltma işlemi
            except Exception as e:
                print(f"Gürültü azaltma işleminde bir hata oluştu: {e}")
                return
            
            try:
                h_esitleme = fonk.histogramEsitleme(gurultuazalt)  # Histogram eşitleme
            except Exception as e:
                print(f"Histogram eşitleme işleminde bir hata oluştu: {e}")
                return
            
            try:
                morfolojik_resim = fonk.morfolojikIslem(h_esitleme)  # Morfolojik işlem
            except Exception as e:
                print(f"Morfolojik işlemde bir hata oluştu: {e}")
                return
            
            try:
                goruntucikarma = fonk.goruntuCikarma(h_esitleme, morfolojik_resim)  # Görüntü çıkarma
            except Exception as e:
                print(f"Görüntü cikarma işleminde bir hata oluştu: {e}")
                return
            
            try:
                goruntuesikleme = fonk.goruntuEsikle(goruntucikarma)  # Görüntü eşikleme
            except Exception as e:
                print(f"Goruntu esikleme isleminde bir hata olustu: {e}")
                return
            
            try:
                cannedge_goruntu = fonk.cannyEdge(goruntuesikleme)  # Canny Edge işlemi
            except Exception as e:
                print(f"Canny Edge isleminde bir hata olustu: {e}")
                return
            
            try:
                gen_goruntu = fonk.genisletmeIslemi(cannedge_goruntu)  # Genişletme işlemi
            except Exception as e:
                print(f"Genisletme isleminde bir hata olustu: {e}")
                return
            
            try:
                screenCnt = fonk.konturIslemi(img, gen_goruntu)  # Kontur işlemi
                if screenCnt is None:
                    print("Kontur bulunamadi, islem sonlandirildi.")
                    return
            except Exception as e:
                print(f"Kontur isleminde bir hata olustu: {e}")
                return
            
            try:
                yeni_goruntu = fonk.maskelemeIslemi(img_gray, img, screenCnt)  # Maskeleme işlemi
                if yeni_goruntu is None:
                    print("Maskeleme işlemi basarisiz, islem sonlandirildi.")
                    return
            except Exception as e:
                print(f"Maskeleme isleminde bir hata olustu: {e}")
                return
            
            try:
                fonk.plakaIyilestir(yeni_goruntu)  # Son iyileştirme işlemi
            except Exception as e:
                print(f"Plaka iyilestirme isleminde bir hata olustu: {e}")
                
    except Exception as e:
        print(f"Beklenmeyen bir hata olustu: {e}")

goruntu()