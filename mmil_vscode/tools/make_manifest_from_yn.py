# make_manifest_from_yn.py (디버그+확장자 인자 지원)
import json, sys, argparse
from pathlib import Path
from collections import Counter, defaultdict

DEFAULT_VIDEO_EXTS = {
    ".mp4",".avi",".mov",".mkv",".m4v",".mpg",".mpeg",
    ".wmv",".webm",".mts",".m2ts",".ts",".3gp",
    # 현장에서 자주 빠지는 것들 추가
    ".3g2",".asf",".mxf",".vob",".mod",".h264",".264",".hevc",".rm",".rmvb",".dat",".dv"
}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

parser = argparse.ArgumentParser()
parser.add_argument("root", help="테스트 루트(안에 Y/N 폴더 존재)")
parser.add_argument("out", help="출력 jsonl 경로")
parser.add_argument("--ext", default=",".join(sorted(x.strip(".") for x in DEFAULT_VIDEO_EXTS)),
                    help="영상 확장자 리스트(쉼표 구분, 예: mp4,mov,avi)")
parser.add_argument("--debug", action="store_true", help="확장자/스킵 샘플 출력")
args = parser.parse_args()

root = Path(args.root)
out  = Path(args.out)

VIDEO_EXTS = set("." + e.strip().lower().lstrip(".") for e in args.ext.split(","))

pairs = [("train-Y", 1), ("train-N", 0)]  # train-Y=낙상, train-N=비낙상

items = []
seen_exts = { sub: Counter() for sub, _ in pairs }
skipped   = { sub: [] for sub, _ in pairs }

for sub, label in pairs:
    folder = root / sub
    if not folder.exists():
        if args.debug: print(f"[warn] 폴더 없음: {folder}")
        continue
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        seen_exts[sub][ext] += 1
        if ext in VIDEO_EXTS:
            # 같은 이름의 오디오 파일이 옆에 있으면 자동 연결(없으면 None)
            audio = None
            for aext in AUDIO_EXTS:
                ap = p.with_suffix(aext)
                if ap.exists():
                    audio = str(ap.resolve())
                    break
            items.append({
                "video": str(p.resolve()),
                "audio": audio,
                "label": label
            })
        else:
            if len(skipped[sub]) < 50:  # 너무 많으면 50개만 샘플
                skipped[sub].append(str(p))

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in items:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(items)} lines -> {out}")
print("Label counts:", Counter(r["label"] for r in items))

if args.debug:
    for sub, _ in pairs:
        total_files = sum(seen_exts[sub].values())
        print(f"\n[{sub}] total files: {total_files}")
        print(f"[{sub}] by extension (top 15):")
        for ext, cnt in seen_exts[sub].most_common(15):
            mark = "" if ext in VIDEO_EXTS else " (SKIP)"
            print(f"  {ext or '<noext>'}: {cnt}{mark}")
        if skipped[sub]:
            print(f"[{sub}] skipped samples (up to 50):")
            for s in skipped[sub]:
                print("  -", s)