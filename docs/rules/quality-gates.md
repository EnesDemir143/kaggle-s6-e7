# Quality Gate Rules

1. Kod değişikliği tamamlanmadan `uv run python scripts/check.py` çalıştırılır.
2. Zorunlu kapılar pytest, Ruff, Mypy ve `python -m compileall -q src tests scripts` kontrolleridir.
3. Bir kapı başarısızsa hata düzeltilip tüm sıra yeniden çalıştırılır.
4. Yeni Python kaynak klasörü tüm kalite kontrollerine eklenir.
5. Notebook değişikliklerinde etkilenen notebook temiz kernel ile çalıştırılır.
