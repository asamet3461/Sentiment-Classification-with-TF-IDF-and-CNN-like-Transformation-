#  Sentiment Classification with TF-IDF and CNN-like Features  
#  TF-IDF ve CNN-benzeri Özelliklerle Duygu Sınıflandırması

##  Description | Açıklama

This project performs sentiment classification on textual data using a manually applied convolution kernel on TF-IDF features and a logistic regression classifier.

Bu proje, metin verisi üzerinde TF-IDF özelliklerine elle uygulanan konvolüsyon çekirdeği ve lojistik regresyon sınıflayıcı kullanarak duygu sınıflandırması yapar.

---

##  Steps | Adımlar

1. **Preprocessing | Ön İşlem**
   - Load CSV dataset  
   - Drop missing values  
   - Apply TF-IDF (top 5000 words)  
   - Encode sentiment labels (positive, neutral, negative)

   - CSV veri kümesi yüklenir  
   - Eksik veriler temizlenir  
   - TF-IDF (en sık geçen 5000 kelime) uygulanır  
   - Duygu etiketleri sayısal olarak kodlanır

2. **Feature Transformation | Özellik Dönüşümü**
   - Text vectors are reshaped into 2D (50x100) to simulate image-like inputs  
   - A simple 2D convolution is manually applied with a kernel  

   - Metin vektörleri 2D (50x100) olarak yeniden şekillendirilir  
   - Basit bir 2D konvolüsyon işlemi elle uygulanır

3. **Classification | Sınıflandırma**
   - Logistic Regression is trained on transformed features  
   - Accuracy is evaluated on test set  

   - Lojistik Regresyon modeli dönüştürülmüş özelliklerle eğitilir  
   - Başarı oranı test verisi üzerinde değerlendirilir

---

##  Output | Çıktı

- Accuracy achieved: **84.75%**

- Elde edilen doğruluk: **%84.75**

---

##  Requirements | Gereksinimler

```bash
pip install numpy pandas scikit-learn
