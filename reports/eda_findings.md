# EDA Findings and Preprocessing Decision Log

Bu dosya notebook çıktılarından doğrulanan bulguların merkezi karar günlüğüdür.

Durum etiketleri: **Gözlem**, **Aday**, **Kabul**, **Red**.

## 1. Veri yapısı

- **Gözlem:** Train 690.088, test 295.753 satırdır.
- **Gözlem:** Target dışındaki train kolonları test ile uyumludur.

## 2. Target dağılımı

- **Gözlem:** `at-risk` baskın sınıftır; `fit` en seyrek sınıftır.
- **Kabul:** Gelecekteki model değerlendirmesinde balanced accuracy ve sınıf bazlı recall izlenecek.

## 3. Missingness

- Notebook 02 çalıştırıldıktan sonra bulgular buraya eklenecek.

## 4. Train-test shift

- Notebook 03 çalıştırıldıktan sonra bulgular buraya eklenecek.

## 5. Outlier analizi

- **Kabul:** İlk aşamada satır silinmeyecek.
- **Aday:** Train quantile eşiklerinden üretilen outlier flag'leri.
- **Aday:** Clipping yalnız Preprocess V3 kapsamında değerlendirilecek.

## 6. Feature engineering adayları

- **Aday:** Missing flag'leri, `missing_count`, ratio feature'lar ve kategorik interaction'lar.
- **Aday:** Sağlık eşiği flag'leri ayrı incelenecek; tıbbi tanı olarak yorumlanmayacak.

## 7. Preprocessing paketleri

| Paket | Tanım | Durum |
|---|---|---|
| V1 | Numeric median, categorical `missing`, missing flag; clipping/silme yok | Aday |
| V2 | V1 + ratio, missing count, kategorik interaction ve outlier flag | Aday/Favori |
| V3 | V2 + train %0.1–%99.9 clipping | Ayrı deney |

## 8. Açık kararlar

- Missingness target sinyali yeterince güçlü mü?
- Hangi ratio ve interaction feature'ları tutulmalı?
- Train-test shift nedeniyle özel handling gereken kolon var mı?
