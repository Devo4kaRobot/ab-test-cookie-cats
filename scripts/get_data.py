"""Скачивает cookie_cats.csv и проверяет контрольную сумму.

Источник данных — открытый датасет «Mobile Games A/B Testing — Cookie Cats» (Kaggle / DataCamp,
данные игры Cookie Cats от Tactile Entertainment). Файл не хранится в репозитории; скрипт берёт его
с публичного зеркала на GitHub и сверяет SHA-256 (тот же файл лежит в нескольких независимых репозиториях).
Можно скачать и вручную с Kaggle — контрольная сумма должна совпасть.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

SHA256 = "5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd"
URLS = [
    "https://raw.githubusercontent.com/zunaira-ahmad/cookie-cats-ab-test/main/cookie_cats.csv",
    "https://raw.githubusercontent.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv",
]
DST = Path(__file__).resolve().parents[1] / "data" / "cookie_cats.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DST.exists() and sha(DST) == SHA256:
        print(f"Уже есть: {DST}"); return
    DST.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            print(f"Скачиваю {url}")
            DST.write_bytes(urllib.request.urlopen(url, timeout=60).read())
            if sha(DST) == SHA256:
                print(f"OK: {DST} (SHA-256 совпадает)"); return
            print("  контрольная сумма не совпала, пробую другое зеркало")
        except Exception as e:  # noqa: BLE001
            print(f"  не получилось: {e}")
    sys.exit("Не удалось скачать. Скачай cookie_cats.csv с Kaggle вручную и положи в data/.")


if __name__ == "__main__":
    main()
