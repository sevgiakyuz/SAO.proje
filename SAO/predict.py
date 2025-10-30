import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

MODEL_KLASORU = "./sao_human_bottleneck_model"

print(f"Eğitilmiş Darboğaz Tespit Modeli ({MODEL_KLASORU}) yükleniyor...")
print("Bu işlem 5-10 saniye sürebilir, lütfen bekleyin...")

try:
    MODEL_PATH = os.path.abspath(MODEL_KLASORU)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
except OSError:
    print(f"HATA: '{MODEL_PATH}' dizininde model bulunamadı.")
    print("Lütfen önce 'train.py' (sao_human_bottleneck_model versiyonu) çalıştırın.")
    exit()

device = torch.device("cpu")
model.to(device)
print("Model başarıyla yüklendi ve CPU üzerinde çalışmaya hazır.")


id2label = {0: "NORMAL SÜREÇ", 1: "İNSAN KAYNAKLI DARBOĞAZ TESPİT EDİLDİ"}


def predict_bottleneck(text):

    "Verilen bir metni analiz eder ve 'DARBOĞAZ' veya 'NORMAL' olarak sınıflandırır (tahmin eder)."

    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = torch.argmax(logits, dim=1).item()

    # Olasılıkları hesapla
    probabilities = F.softmax(logits, dim=1).squeeze()
    confidence = probabilities[predicted_class_id].item()

    label = id2label[predicted_class_id]

    return label, confidence


# Ana Uygulama Döngüsü
print("\n--- SAO İnsan Kaynaklı Darboğaz Tahmin Sistemi ---")
print("Model, girdiğiniz sürecin bir darboğaz olup olmadığını 'tahmin' edecektir.")
print("(Çıkmak için 'çıkış' yazıp Enter'a basın)\n")

while True:
    user_input = input("Lütfen durumu (log, e-posta, süreç notu) girin: ")

    if user_input.lower() in ['çıkış', 'cikis', 'exit', 'quit']:
        print("Programdan çıkılıyor... Hoşçakalın.")
        break

    if not user_input.strip():
        print("Lütfen geçerli bir metin girin.")
        continue

    print("\n>>> Süreç analiz ediliyor, lütfen bekleyin...")
    try:
        etiket, skor = predict_bottleneck(user_input)
        print("\nMODEL TAHMİNİ:")
        print(f" Durum: {etiket} (Güven Skoru: %{skor * 100:.2f})\n")
    except Exception as e:
        print(f"Model tahmin yaparken bir hata oluştu: {e}")

    print("-" * 70)