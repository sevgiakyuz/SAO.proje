import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
print(f"'{MODEL_NAME}' modeli (Sınıflandırıcı) yükleniyor...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
except Exception as e:
    print(f"Hata: Model yüklenemedi. İnternet bağlantınızı kontrol edin. Hata: {e}")
    exit()

print("Model ve tokenizer başarıyla yüklendi.")

DATA_FILE = "darbogaz_verileri_3k.csv"
try:
    df = pd.read_csv(DATA_FILE)
    print(f"'{DATA_FILE}' dosyasından {len(df)} satır veri yüklendi.")
except FileNotFoundError:
    print(f"Hata: '{DATA_FILE}' dosyası bulunamadı. Lütfen önce 'data_generator.py'yi çalıştırın.")
    exit()

df = df.rename(columns={'metin': 'text', 'etiket': 'label'})
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
print(f"Veri {len(train_dataset)} eğitim, {len(test_dataset)} test satırına ayrıldı.")



def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)


print("Veri seti tokenize ediliyor...")

tokenized_train_ds = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_test_ds = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
print("Tokenizasyon tamamlandı.")



def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


OUTPUT_DIR = "./sao_human_bottleneck_model"


ADIM_SAYISI_PER_EPOCH = len(train_dataset) // 8

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=150,
    eval_steps=ADIM_SAYISI_PER_EPOCH,
    save_steps=ADIM_SAYISI_PER_EPOCH,


    fp16=torch.cuda.is_available(),
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("--- BERT Darboğaz Tespit Eğitimi Başlıyor (3000 Veri) ---")
trainer.train()
print("--- Model Eğitimi Tamamlandı ---")

model_save_path = "./sao_human_bottleneck_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Eğitilmiş DARBOĞAZ TESPİT MODELİ '{model_save_path}' dizinine kaydedildi.")

eval_results = trainer.evaluate()
print(f"Değerlendirme sonuçları (F1 Score): {eval_results['eval_f1']:.4f}")