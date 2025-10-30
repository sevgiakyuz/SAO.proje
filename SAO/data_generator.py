import pandas as pd
import random

# İNSAN KAYNAKLI DARBOĞAZ ŞABLONLARI (Etiket = 1)
templates_darbogaz = [
    "{kisi} {zaman} {belge} onayını bekletiyor.",
    "{departman} ekibinden {zaman} geri dönüş alınamadı, süreç durdu.",
    "{kisi}, {zaman} yapılması gereken toplantıyı erteledi, {sonuc}.",
    "Müşteri verisi {kisi} tarafından {zaman} manuel olarak girilmedi.",
    "Destek bileti #{id}, {kisi} tarafından {zaman} incelenmedi.",
    "{departman} planlaması {zaman} yapılmadı, kaynak ataması bekliyor.",
    "Raporlama {kisi} tarafından {zaman} tamamlanmadı, {sonuc}."
]

# NORMAL SÜREÇ ŞABLONLARI (Etiket = 0)
templates_normal = [
    "{belge} {kisi} tarafından zamanında onaylandı.",
    "{departman} ekibi {zaman} içinde geri dönüş sağladı.",
    "Haftalık planlama toplantısı {kisi} ile başarıyla yapıldı.",
    "Müşteri verisi sisteme otomatik olarak aktarıldı.",
    "Destek bileti #{id} {kisi} tarafından çözüldü ve kapatıldı.",
    "{departman} planlaması tamamlandı, kaynaklar atandı.",
    "Raporlama {kisi} tarafından yönetime sunuldu."
]

kisiler = ['Yönetici', 'Ekip Lideri', 'Departman Müdürü', 'Ahmet Y.', 'Ayşe K.']
departmanlar = ['Finans', 'İK', 'Operasyon', 'Satış', 'IT Destek']
zamanlar = ['3 gündür', '48 saattir', '1 haftadır', '5 iş günüdür', 'hala']
belgeler = ['bütçe', 'fatura', 'proje planı', 'izin talebi']
sonuclar = ['işlem bekliyor', 'proje gecikiyor', 'müşteri bekliyor']

data = []
TARGET_COUNT_PER_CLASS = 1500
TOTAL_COUNT = TARGET_COUNT_PER_CLASS * 2

print(f"{TOTAL_COUNT} adet (İnsan Kaynaklı) darboğaz verisi üretiliyor...")

# 1. Darboğaz Verisi Üretimi (Etiket 1)
for _ in range(TARGET_COUNT_PER_CLASS):
    template = random.choice(templates_darbogaz)
    metin = template.format(
        id=random.randint(1000, 9999),
        kisi=random.choice(kisiler),
        zaman=random.choice(zamanlar),
        departman=random.choice(departmanlar),
        belge=random.choice(belgeler),
        sonuc=random.choice(sonuclar)
    )
    data.append({'metin': metin, 'etiket': 1})

# 2. Normal Veri Üretimi (Etiket 0)
for _ in range(TARGET_COUNT_PER_CLASS):
    template = random.choice(templates_normal)
    metin = template.format(
        id=random.randint(1000, 9999),
        kisi=random.choice(kisiler),
        zaman=random.choice(['bugün', 'zamanında', '1 saat içinde']),
        departman=random.choice(departmanlar),
        belge=random.choice(belgeler)
    )
    data.append({'metin': metin, 'etiket': 0})

print("Veri üretimi tamamlandı.")
random.shuffle(data)
df = pd.DataFrame(data)

# Yeni veri dosyamızın adı
output_filename = 'darbogaz_verileri_3k.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"Toplam {len(df)} satırlık veri '{output_filename}' dosyasına kaydedildi.")
print(df.head())