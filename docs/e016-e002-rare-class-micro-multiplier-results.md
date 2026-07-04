# E016 — E002 Rare-Class Mikro Multiplier Pipeline ve Sonuç Raporu

## 1. Amaç

E016, blend veya yeni model kullanmadan yalnız E002 probability'leri üzerinde küçük
`fit` ve `unhealthy` multiplier artışlarını sınar. Amaç E002'nin güçlü karar sınırını
bozmadan, ölçülebilir fakat küçük bir rare-class düzeltmesi bulmaktır.

Kaynaklar:

```text
outputs/experiments/E002/oof_proba.npy
outputs/experiments/E002/test_proba.npy
```

E002 taban multiplier:

```text
at-risk   = 0.1892298422
fit       = 1.4444532324
unhealthy = 1.3663169255
```

## 2. Seçim politikası

Beş adayın her biri OOF üzerinde skorlanır ve E002 tuned test tahminleriyle label
disagreement oranı hesaplanır. Bir aday ancak üç koşulu birlikte sağlarsa submit adayıdır:

```text
OOF balanced accuracy >= 0.94855
E002 disagreement     >= 0.03%
E002 disagreement     <= 0.12%
```

Bandın config karşılığı fraction olarak `[0.0003, 0.0012]` değeridir. Birden fazla aday
geçerse en yüksek OOF skorlu aday seçilir. Hiçbiri geçmezse `submission.csv` yazılmaz.

Bu politika önemlidir: yalnız OOF skoru yüksek fakat E002 ile pratik olarak aynı olan bir
dosya son submission hakkını tüketmemelidir.

## 3. Pipeline akışı

1. E002 OOF/test probability, ID, label mapping ve multiplier artefaktlarını yükle.
2. E002 tuned test label'larını disagreement referansı olarak üret.
3. A–E multiplier çiftlerinin her birini OOF ve test probability'lerine uygula.
4. OOF balanced accuracy, test disagreement ve değişen test satırı sayısını hesapla.
5. OOF ve disagreement filtrelerini aday bazında uygula.
6. Geçen aday varsa en yüksek OOF skorunu seçip tek submission yaz.
7. Geçen aday yoksa yalnız config, metrics, eligibility ve toplu rapor yaz.

Test etiketleri veya leaderboard etiketi kullanılmaz. Test probability'lerinden yalnız
etiketsiz disagreement ölçüsü hesaplanır.

## 4. Adaylar ve sonuçlar

| Aday | Fit scale | Unhealthy scale | OOF bal. acc. | E002 disagreement | Değişen test satırı | OOF filtresi | Disagreement filtresi | Sonuç |
|---|---:|---:|---:|---:|---:|---|---|---|
| A | 1.0025 | 1.0025 | 0.948570818771 | %0.00338 | 10 | PASS | FAIL | Elendi |
| B | 1.0050 | 1.0050 | 0.948570704158 | %0.00575 | 17 | PASS | FAIL | Elendi |
| C | 1.0075 | 1.0075 | **0.948578814818** | **%0.01014** | **30** | PASS | FAIL | Elendi |
| D | 1.0050 | 1.0000 | 0.948574607175 | %0.00203 | 6 | PASS | FAIL | Elendi |
| E | 1.0000 | 1.0050 | 0.948563518012 | %0.00507 | 15 | PASS | FAIL | Elendi |

Beş adayın tamamı `0.94855` OOF tabanını geçti. Ancak en farklı aday C bile E002'nin
yalnız 30 test satırını, yani `%0.01014` oranını değiştirdi. Bu değer minimum `%0.03`
eşiğinin yaklaşık üçte biridir. Dolayısıyla hiçbir aday iki filtreyi birlikte geçemedi.

## 5. Nihai karar

```text
selected_candidate = null
eligible            = false
submission.csv      = üretilmedi
final recommendation = E002 tuned
```

OOF bakımından en iyi aday C olsa da seçim kuralı bilinçli olarak yalnız “en yüksek OOF”
dememektedir. C'nin disagreement seviyesi gereken minimum müdahaleyi sağlamadığı için son
submit hakkını kullanmaya değer yeni bir karar sınırı üretmemiştir.

Bu sonuç E002 çevresindeki `+0.25%–+0.75%` rare-class multiplier artışlarının label
seviyesinde aşırı küçük kaldığını gösterir. Filtreyi sonradan gevşetmek deney protokolünü
leaderboard sonucuna göre değiştirmek olur; bu nedenle önerilmez.

## 6. Çalıştırma

```bash
bash scripts/run_e016_experiment.sh
```

Yeniden değerlendirme:

```bash
FORCE=1 bash scripts/run_e016_experiment.sh
```

Script submission oluşup oluşmadığını kontrol eder. Bu çalışmada beklenen terminal mesajı:

```text
No E016 candidate passed all filters; no submission was produced. Keep E002 final.
```

## 7. Dosyalar

Pipeline tanımı:

```text
configs/e016_experiment.yaml
scripts/run_e016_experiment.sh
```

Üretilen karar artefaktları:

```text
outputs/experiments/E016_E002_rare_class_tiny_boost/config.json
outputs/experiments/E016_E002_rare_class_tiny_boost/metrics.json
outputs/experiments/E016_E002_rare_class_tiny_boost/eligibility.json
outputs/experiments/e016_eligibility_report.csv
outputs/experiments/e016_eligibility_report.json
```

`metrics.json` beş adayın skorlarını, multiplier'larını, disagreement değerlerini ve
filtre kararlarını saklar. `submission.csv` dosyasının yokluğu hata değil, filtre
politikasının doğru uygulanmasının sonucudur.

## 8. Sonuç

E016 planlandığı biçimde çalıştırıldı fakat submit edilebilir aday üretmedi. E002-only
mikro multiplier yaklaşımı OOF tabanını korudu; buna karşılık E002'den yeterince farklı
bir test kararı oluşturamadı. Son hak kullanılmamalı ve **E002 tuned final submission**
olarak korunmalıdır.
