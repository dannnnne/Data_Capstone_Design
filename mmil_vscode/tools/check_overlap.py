import json
from pathlib import Path

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

root = Path("C:/Users/User/Desktop/경희대/데캡디/mmil_vscode")

train_items = read_jsonl(root / "data" / "train_manifest.jsonl")
test_items  = read_jsonl(root / "data" / "test_manifest.jsonl")

train_videos = {it["video"] for it in train_items}
test_videos  = {it["video"] for it in test_items}

inter = train_videos & test_videos

print("train video 수:", len(train_videos))
print("test video 수:", len(test_videos))
print("겹치는 video 수:", len(inter))

# 혹시 겹치면 몇 개만 프린트
for p in list(inter)[:20]:
    print("  overlap:", p)
