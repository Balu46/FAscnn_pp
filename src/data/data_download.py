# -------------------------------------------
import os
import argparse
import requests
import zipfile
import tarfile
from tqdm import tqdm

# -------------------------------------------
# KONFIGURACJA
# -------------------------------------------
DATA_DIR = "cityscapes"
ARCHIVE = "cityscapes_download"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/wxxwm55olryrgf3pcmt8r/gtFine_trainvaltest.zip?rlkey=znla88tz77lrby85p3z1y6w8m&st=h7zhvmtu&dl=1    "  # link do pliku .zip lub .tar.gz z Dropboxa

IMAGENET_DIR = "imagenet"
IMAGENET_ARCHIVE = "imagenet_download"
IMAGENET_URL = ""  # wstaw link do archiwum .zip lub .tar.gz
IMAGENET_EXPECTED_SUBDIRS = ["train", "val"]
# -------------------------------------------


def download_file(url, out_path):
    """Pobiera plik z paskiem postępu i naprawia linki Dropboxa."""
    print(f"[INFO] Pobieram dane z: {url}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")

        # Dropbox zwrócił HTML → popraw link
        if "text/html" in content_type:
            print("[WARN] Otrzymano HTML zamiast pliku – poprawiam link...")
            if "dl=0" in url:
                url = url.replace("dl=0", "dl=1")
                print(f"[INFO] Nowy link: {url}")
                return download_file(url, out_path)
            else:
                raise ValueError("Dropbox zwrócił stronę HTML – podaj bezpośredni link do pliku.")

        # Pobieranie z paskiem
        total = int(r.headers.get("Content-Length", 0))
        block_size = 8192
        progress = tqdm(total=total, unit="B", unit_scale=True, desc="Pobieranie", ascii=True)

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        progress.close()

    print("[INFO] Pobieranie zakończone.")


def extract_archive(path, dest):
    """Rozpakowuje .zip lub .tar.gz"""
    print("[INFO] Rozpakowuję archiwum...")

    if path.endswith(".zip"):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(dest)

    elif path.endswith(".tar.gz") or path.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tar_ref:
            tar_ref.extractall(dest)

    else:
        raise ValueError("Obsługuję tylko pliki .zip i .tar.gz")

    print("[INFO] Rozpakowywanie zakończone.")


def archive_path_from_url(url, archive_base):
    """Zwraca nazwę archiwum na podstawie URL."""
    if ".zip" in url:
        return archive_base + ".zip"
    if ".tar.gz" in url or ".tgz" in url:
        return archive_base + ".tar.gz"
    raise ValueError("Link musi wskazywać na ZIP lub TAR.GZ")


def cityscapes_exists():
    """Sprawdza czy dane Cityscapes są już rozpakowane."""
    return (
        os.path.isdir(os.path.join(DATA_DIR, "leftImg8bit")) and
        os.path.isdir(os.path.join(DATA_DIR, "gtFine"))
    )


def imagenet_exists():
    """Sprawdza czy dane ImageNet są już rozpakowane."""
    if not os.path.isdir(IMAGENET_DIR):
        return False
    if not IMAGENET_EXPECTED_SUBDIRS:
        return any(os.scandir(IMAGENET_DIR))
    return all(os.path.isdir(os.path.join(IMAGENET_DIR, subdir)) for subdir in IMAGENET_EXPECTED_SUBDIRS)


def download_and_extract(url, archive_base, dest_dir, exists_fn, label, require_url=True):
    """Pobiera i rozpakowuje wskazany dataset."""
    url = (url or "").strip()
    print(f"[INFO] {label} URL: {url if url else 'BRAK (uzupełnij w data_download.py)'}")

    if exists_fn():
        print(f"[INFO] {label} już istnieje — pomijam pobieranie.")
        return

    if not url:
        if require_url:
            raise ValueError(f"Brak URL dla {label}. Uzupełnij zmienną w data_download.py.")
        print(f"[WARN] Brak URL dla {label} — pomijam pobieranie. Uzupełnij IMAGENET_URL w data_download.py.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    archive_path = archive_path_from_url(url, archive_base)

    if not os.path.isfile(archive_path):
        download_file(url, archive_path)
    else:
        print("[INFO] Archiwum istnieje — pomijam pobieranie.")

    extract_archive(archive_path, dest_dir)

    if exists_fn():
        print(f"[INFO] {label} gotowe!")
    else:
        print(f"[ERROR] Struktura plików dla {label} nie wygląda poprawnie.")


def parse_args():
    parser = argparse.ArgumentParser(description="Pobieranie datasetów (Cityscapes / ImageNet).")
    parser.add_argument(
        "--only",
        default="",
        help="Lista datasetów do pobrania (oddzielone przecinkami): cityscapes,imagenet. Domyślnie oba.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    only = [item.strip().lower() for item in args.only.split(",") if item.strip()]
    valid = {"cityscapes", "imagenet"}
    unknown = [item for item in only if item not in valid]
    if unknown:
        raise ValueError(f"Nieznane wartości w --only: {', '.join(unknown)}. Dozwolone: cityscapes, imagenet.")

    use_cityscapes = not only or "cityscapes" in only
    use_imagenet = not only or "imagenet" in only

    if use_cityscapes:
        download_and_extract(
            DROPBOX_URL,
            ARCHIVE,
            DATA_DIR,
            cityscapes_exists,
            "Cityscapes",
            require_url=True,
        )

    if use_imagenet:
        download_and_extract(
            IMAGENET_URL,
            IMAGENET_ARCHIVE,
            IMAGENET_DIR,
            imagenet_exists,
            "ImageNet",
            require_url=False,
        )


if __name__ == "__main__":
    main()
