import os
from pathlib import Path



# 실행 예시
# bash
# python encoding_scan.py C:\Users\ngins\Git\test\dataset_yolo\test\labels



# Optional encoding detectors
try:
    from charset_normalizer import from_bytes as cn_from_bytes
    HAS_CN = True
except ImportError:
    HAS_CN = False

try:
    import chardet
    HAS_CH = True
except ImportError:
    HAS_CH = False

# Common text extensions
TEXT_EXTS = {
    ".txt", ".csv", ".tsv", ".yaml", ".yml", ".json", ".xml",
    ".md", ".ini", ".cfg", ".log", ".py", ".html", ".js", ".css"
}

# BOM signatures
BOMS = {
    "UTF-8": b"\xef\xbb\xbf",
    "UTF-16-LE": b"\xff\xfe",
    "UTF-16-BE": b"\xfe\xff",
    "UTF-32-LE": b"\xff\xfe\x00\x00",
    "UTF-32-BE": b"\x00\x00\xfe\xff",
}

def has_bom(data: bytes) -> bool:
    return any(data.startswith(sig) for sig in BOMS.values())

def detect_encoding(data: bytes):
    bom = has_bom(data)

    # charset-normalizer
    if HAS_CN:
        try:
            result = cn_from_bytes(data)
            best = result.best()
            if best and best.encoding:
                enc = best.encoding
                conf = getattr(best, "confidence", None)
                return enc, conf, bom
        except Exception:
            pass

    # chardet fallback
    if HAS_CH:
        try:
            result = chardet.detect(data)
            enc = result.get("encoding")
            conf = result.get("confidence")
            return enc, conf, bom
        except Exception:
            pass

    # Manual fallback
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "shift_jis", "windows-1252"]:
        try:
            data.decode(enc)
            return enc, None, bom
        except Exception:
            continue

    return None, None, bom

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def scan_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            path = Path(dirpath) / fname
            if path.suffix.lower() in TEXT_EXTS:
                yield path

def print_encoding_table(root: Path):
    print(f"{'Encoding':<20} {'Conf':>6} {'BOM':>5}  {'Size':>9}  Path")
    print("-" * 80)

    for path in scan_files(root):
        try:
            raw = path.read_bytes()
            enc, conf, bom = detect_encoding(raw)

            # ASCII 보정
            if enc == "ascii":
                enc = "utf-8 (ascii subset)"

            enc_str = enc if enc else "unknown"
            conf_str = f"{conf*100:5.1f}%" if isinstance(conf, (int, float)) else "  n/a"
            bom_str = "yes" if bom else " no"
            size_str = human_size(path.stat().st_size)

            print(f"{enc_str:<20} {conf_str:>6} {bom_str:>5}  {size_str:>9}  {path}")
        except Exception as e:
            print(f"{'error':<20} {'  n/a':>6} {' n/a':>5}  {'     ?':>9}  {path} -> {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python encoding_scan.py [경로]")
    else:
        root_path = Path(sys.argv[1]).expanduser().resolve()
        if not root_path.exists():
            print(f"경로가 존재하지 않습니다: {root_path}")
        else:
            print_encoding_table(root_path)
