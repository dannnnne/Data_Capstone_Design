# tools/build_manifest_from_TS_TL.py
import os, re, json, argparse
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

# ex) 00275_H_D_BY_C8 -> 00275_H_D_BY
CAM_SUFFIX = re.compile(r"^(.*)_C[0-9A-Za-z]+$")

def canonical_id(name: str) -> str:
    m = CAM_SUFFIX.match(name)
    return m.group(1) if m else name

def find_videos(ts_root: Path, camera_filter=None):
    vids = []
    for p in ts_root.rglob("*"):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            stem = p.stem
            # 카메라 필터: 예) ['C8','C4']
            if camera_filter:
                # 파일명이 ..._C8 형태인지 확인
                cam = stem.split("_")[-1]  # C8
                if cam not in camera_filter:
                    continue
            vids.append(p)
    return sorted(vids)

def extract_label_from_json_text(text_lower: str) -> int:
    """
    JSON 구조를 몰라도 텍스트 전체에서 키워드로 낙상(1)/비낙상(0) 추정.
    필요하면 여기 규칙을 더 추가.
    """
    # 확실 키워드 우선
    if "fall" in text_lower or "낙상" in text_lower:
        return 1
    # 흔한 이진 라벨 표현들
    positives = ["abnormal", "positive", '"label":1', '"class":1', '"fall":1']
    negatives = ["normal", "negative", '"label":0', '"class":0', '"fall":0']
    if any(k in text_lower for k in positives):
        return 1
    if any(k in text_lower for k in negatives):
        return 0
    # 모호하면 0
    return 0

def load_label_map_from_TL(tl_root: Path) -> dict:
    """
    TL 아래의 */<name>/<name>.json 들을 훑어
    base_id(카메라 서픽스 제거) -> 0/1 맵 생성.
    충돌 시 양성(1) 우선 합산.
    """
    lab = {}
    for p in tl_root.rglob("*.json"):
        if not p.is_file():
            continue
        stem = p.stem
        base = canonical_id(stem)
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            try:
                txt = p.read_text(encoding="cp949", errors="ignore").lower()
            except Exception:
                txt = ""
        y = extract_label_from_json_text(txt)
        # 여러 json이 같은 base를 가리키면 1이 하나라도 있으면 1
        lab[base] = max(y, lab.get(base, 0))
    return lab

def guess_from_filename(path: Path) -> int:
    s = path.stem.lower()
    return 1 if ("fall" in s or "낙상" in s) else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True,
                    help="...\\1.데이터\\Training 폴더 경로")
    ap.add_argument("--out", required=True,
                    help="생성할 train_manifest.jsonl 경로")
    ap.add_argument("--ts-name", default="01.원천데이터",
                    help="원천데이터 폴더명 (기본: 01.원천데이터)")
    ap.add_argument("--tl-name", default="02.라벨링데이터",
                    help="라벨링데이터 폴더명 (기본: 02.라벨링데이터)")
    ap.add_argument("--camera-filter", default=None,
                    help="포함할 카메라 코드 CSV (예: C8 또는 C8,C4). 미지정 시 전체 포함")
    ap.add_argument("--force-audio-null", action="store_true",
                    help="항상 audio=null 로 기록 (mp4에서 자동 추출)")
    args = ap.parse_args()

    train_root = Path(args.train_root)
    ts_root = train_root / args.ts_name  # e.g., Training/01.원천데이터
    tl_root = train_root / args.tl_name  # e.g., Training/02.라벨링데이터

    # 하위에 TS / TL 같은 추가 Depth가 있을 수 있으므로, 직접 TS/TL을 지정하지 않고 전체 스캔
    # 다만 원천은 TS 트리, 라벨은 TL 트리 안에서 rglob 사용
    ts_dir = None
    tl_dir = None
    for d in ts_root.rglob("*"):
        if d.is_dir() and d.name.upper() == "TS":
            ts_dir = d
            break
    for d in tl_root.rglob("*"):
        if d.is_dir() and d.name.upper() == "TL":
            tl_dir = d
            break

    if ts_dir is None:
        # TS가 명시 폴더명으로 없을 수도 있으므로, 원천 루트 자체를 사용
        ts_dir = ts_root
    if tl_dir is None:
        tl_dir = tl_root

    camera_filter = None
    if args.camera_filter:
        camera_filter = [c.strip() for c in args.camera_filter.split(",") if c.strip()]

    print(f"[INFO] 영상 탐색: {ts_dir}")
    vids = find_videos(ts_dir, camera_filter=camera_filter)
    if not vids:
        raise SystemExit("[ERR] 영상(.mp4 등)이 발견되지 않았습니다.")

    print(f"[INFO] 라벨 로딩: {tl_dir}")
    label_map = load_label_map_from_TL(tl_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = n_unmatched = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for v in vids:
            stem = v.stem
            base = canonical_id(stem)
            y = label_map.get(base)
            if y is None:
                y = guess_from_filename(v)
                n_unmatched += 1
            y = int(y)

            rec = {
                "video": str(v),
                "audio": None if args.force_audio_null else None,  # 별도 wav 없으면 None 유지
                "label": y
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if y == 1: n_pos += 1
            else: n_neg += 1

    print(f"[DONE] {out_path} 생성")
    print(f" - 총 영상: {len(vids)}개 (pos={n_pos}, neg={n_neg}, 라벨미매칭={n_unmatched})")
    if n_unmatched > 0:
        print("   일부 영상은 파일명 기반(키워드)으로 라벨을 추정했습니다. 필요 시 규칙 보강 권장.")

if __name__ == "__main__":
    main()
