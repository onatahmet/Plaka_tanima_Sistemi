import sqlite3

VERITABANI_ADI = "plaka_kayit.db"  # Tek bir değişken ile veritabanı adı belirleyelim

def baglanti_ac():
    """Veritabanı bağlantısını açmak için yardımcı fonksiyon."""
    return sqlite3.connect(VERITABANI_ADI)

def veritabani_olustur():
    """Veritabanı ve tabloyu oluşturma."""
    try:
        conn = baglanti_ac()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plakalar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plaka TEXT NOT NULL,
                zaman TEXT NOT NULL
            )
        ''')

        conn.commit()
        print("Veritabani ve tablo oluşturuldu.")
    except sqlite3.Error as e:
        print(f"Hata: {e}")
    finally:
        conn.close()

def plaka_ekle(plaka, zaman):
    """Yeni plaka kaydını veritabanına ekleme."""
    try:
        conn = baglanti_ac()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO plakalar (plaka, zaman) VALUES (?, ?)", (plaka, zaman))
        conn.commit()
        print(f"Plaka {plaka} kaydedildi.")
    except sqlite3.Error as e:
        print(f"Hata: {e}")
    finally:
        conn.close()

def plakalari_listele():
    """Veritabanındaki plakaları listeleme."""
    try:
        conn = baglanti_ac()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM plakalar")
        kayitlar = cursor.fetchall()

        if kayitlar is None or not kayitlar:
            return []  # Boş liste döndür

        plakalar_json = [{"plaka_metni": kayit[1], "tarih_saat": kayit[2]} for kayit in kayitlar]

        return plakalar_json

    except sqlite3.Error as e:
        print(f"Hata: {e}")
        return []  # Hata durumunda boş liste döndür
    finally:
        conn.close()







