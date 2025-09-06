import os
import re
#from collections import defaultdict
import itertools



# 경로 설정
target_dir = r"C:/Users/ngins/Downloads/[origin]CarNumber_OCR_data"  # 원본 폴더
source_dir = r"C:/Users/ngins/Git/python.initial/20250904/runs/cropped_images"  # 작업 폴더

IMG_EXTS = (".jpg", ".jpeg", ".png")

# target 매핑: (앞번호, 뒷번호 전체) -> 한글
mapping = {}

for tf in os.listdir(target_dir):
    if not tf.lower().endswith(IMG_EXTS):
        continue
    name, _ = os.path.splitext(tf)
    # 한글 1글자 기준 분리 (뒷번호에 _숫자 포함 가능)
    m = re.match(r"^(\d+)([가-힣])(\d+(?:-\d+)?)$", name)
    if m:
        front_num = m.group(1)   # 예: 01
        hangul_ch = m.group(2)   # 예: 가
        back_num = m.group(3)    # 예: 0107 또는 0107_2
        mapping[(front_num, back_num)] = hangul_ch

# mapping의 앞 5개만 출력
# for k, v in itertools.islice(mapping.items(), 5):
for k, v in mapping.items():
    if k == ('04','0865-2'):
        print(k, "→", v)

# source 처리
for sf in os.listdir(source_dir):
    if not sf.lower().endswith(IMG_EXTS):
        continue
    name, ext = os.path.splitext(sf)
    if "_" not in name:
        continue
    left_part = name.split("_")[0]  # 예: 01-0107_2
    if "-" not in left_part:
        continue
    s_front, s_back = left_part.split("-", 1)

    key = (s_front, s_back)
    if key in mapping:
        hangul_ch = mapping[key]
        print(f"⚠ hangul_ch: {hangul_ch} key:{key}")
        new_name = name.replace("-", hangul_ch, 1) + ext
        old_path = os.path.join(source_dir, sf)
        new_path = os.path.join(source_dir, new_name)
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"변경: {sf} → {new_name} key:{key}")
        else:
            print(f"⚠ 이름 충돌: {new_name} key:{key}")
    else:
        print(f"❌ 매칭 없음: {sf} key:{key}")

print("✅ 처리 완료")
