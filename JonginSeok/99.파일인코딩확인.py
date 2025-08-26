import os
import sys
import argparse
from pathlib import Path


"""
사용 예
기본 확장자 전체 스캔: python detect_encoding.py /path/to/dir

특정 확장자만: python detect_encoding.py /path/to/dir --ext txt,yaml,csv

더 많은 바이트 분석: python detect_encoding.py /path/to/dir --bytes 8388608

필요하면 결과를 CSV로 출력하거나, 바이너리 파일을 제외하는 추가 규칙도 넣어줄 수 있어. 
"""

# Optional detectors
try:
    from charset_normalizer import from_bytes as cn_from_bytes
    HAS_CHARSET_NORMALIZER = True
except Exception:
    HAS_CHARSET_NORMALIZER = False

try:
    import chardet
    HAS_CHARDET = True
except Exception:
    HAS_CHARDET = False


TEXT_EXTS_DEFAULT = {
    ".txt", ".yaml", ".yml", ".csv", ".tsv", ".json",
    ".md", ".xml", ".ini", ".cfg", ".log",
    ".py", ".js", ".html", ".css"
}

# Common encodings to try if no detector available
FALLBACK_ENCODINGS = [
    "utf-8-sig", "utf-8", "cp949", "euc-kr",
    "windows-1252", "iso-8859-1", "shift_jis", "gb18030"
]

BOMS = {
    "UTF-8": b"\xef\xbb\xbf",
    "UTF-16-LE": b"\xff\xfe",
    "UTF-16-BE": b"\xfe\xff",
    "UTF-32-LE": b"\xff\xfe\x00\x00",
    "UTF-32-BE": b"\x00\x00\xfe\xff",
}

def has_bom(raw: bytes) -> bool:
    for sig in BOMS.values():
        if raw.startswith(sig):
            return True
    return False

def detect_encoding(raw: bytes):
    """
    Return tuple: (encoding: str|None, confidence: float|None, bom: bool)
    confidence is 0..1 if available.
    """
    bom = has_bom(raw)

    # 1) charset-normalizer
    if HAS_CHARSET_NORMALIZER:
        try:
            result = cn_from_bytes(raw)
            best = result.best()
            if best is not None and best.encoding:
                conf = getattr(best, "confidence", None)
                # charset-normalizer confidence may be 0..1; ensure clamp
                if isinstance(conf, (int, float)):
                    conf = max(0.0, min(1.0, float(conf)))
                return best.encoding, conf, bom
        except Exception:
            pass

    # 2) chardet
    if HAS_CHARDET:
        try:
            det = chardet.detect(raw)  # {'encoding': 'utf-8', 'confidence': 0.99, ...}
            enc = det.get("encoding")
            conf = det.get("confidence")
            if enc:
                return enc, (float(conf) if conf is not None else None), bom
        except Exception:
            pass

    # 3) Fallback trial decodes
    for enc in FALLBACK_ENCODINGS:
        try:
            raw.decode(enc)
            return enc, None, bom
        except Exception:
            continue

    return None, None, bom

def is_text_ext(path: Path, include_exts):
    return path.suffix.lower() in include_exts

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"

def scan_dir(root: Path, include_exts):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            if is_text_ext(p, include_exts):
                files.append(p)
    return files

def main():
    parser = argparse.ArgumentParser(
        description="List encodings for text files recursively."
    )
    parser.add_argument("path", help="Root directory to scan")
    parser.add_argument(
        "--ext",
        help="Comma-separated list of extensions to include (e.g., txt,yaml,csv). Defaults to common text types.",
        default=None,
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=4_194_304,  # read up to 4MB
        help="Max bytes to read per file for detection (default: 4MB)"
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        sys.exit(1)

    include_exts = TEXT_EXTS_DEFAULT
    if args.ext:
        include_exts = {"." + e.strip().lstrip(".").lower() for e in args.ext.split(",") if e.strip()}

    files = scan_dir(root, include_exts)

    # Header
    print(f"{'Encoding':<14} {'Conf':>6} {'BOM':>5}  {'Size':>9}  Path")
    print("-" * 80)

    for p in sorted(files):
        try:
            raw = p.open("rb").read(args.bytes)
            enc, conf, bom = detect_encoding(raw)
            size = p.stat().st_size
            conf_str = f"{conf*100:5.1f}%" if isinstance(conf, (int, float)) else "  n/a"
            bom_str = "yes" if bom else " no"
            enc_str = enc if enc else "unknown"
            print(f"{enc_str:<14} {conf_str:>6} {bom_str:>5}  {human_size(size):>9}  {p}")
        except Exception as e:
            print(f"{'error':<14} {'  n/a':>6} {' n/a':>5}  {human_size(0):>9}  {p}  -> {e}")

if __name__ == "__main__":
    main()
