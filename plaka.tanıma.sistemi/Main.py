import islem2
print("""        
##############################################                                            # 
#                                            #
#                                            #
#----------- PLAKA TANIMA SİSTEMİNE----------#
#                                            #
#                                            #
##############################################""")

print("Resimden plaka tanima işlemi yapmak için 1'e basiniz")

adimleri_goster=False
try:
    secim = int(input("Seçiminiz:"))

    if secim == 1:
        islem2.goruntu()
 
    else:
        exit()

except ValueError:
    print("Hatali seçim Lütfen menüden seçim yapiniz.") 