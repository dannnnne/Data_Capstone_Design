# -*- coding: utf-8 -*-
# 평가 스크립트: MultiModalMILNet + MIL 풀링으로 bag 단위 지표 계산
import argparse, json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# 지표
try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
except Exception as e:
    raise SystemExit("scikit-learn이 필요합니다. pip install scikit-learn")

# 프로젝트 모듈 (네 코드 그대로 사용)
from mmil.config import load_config
from mmil.data import MultiModalMILDataset, collate_fn
from mmil.model import MultiModalMILNet

# ---------- MIL bag pooling (학습과 동일 옵션) ----------
def _ensure_bool(mask: torch.Tensor) -> torch.Tensor:
    return mask if mask.dtype == torch.bool else mask.bool()

@torch.no_grad()
def apply_mil_pool(logits: torch.Tensor, mask: torch.Tensor,
                   pool: str = "lse", tau: float = 5.0, topk_ratio: float = 0.1) -> torch.Tensor:
    """
    logits: (B,T), mask: (B,T)->bool
    반환: bag_logit (B,)
    """
    mask = _ensure_bool(mask)
    B, T = logits.shape
    if pool == "mean":
        w = mask.float()
        bag = logits.masked_fill(~mask, 0).sum(1) / w.sum(1).clamp_min(1.0)
        return bag
    elif pool == "max":
        x = logits.masked_fill(~mask, float("-inf"))
        out = x.max(1).values
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    elif pool == "topk":
        x = logits.masked_fill(~mask, float("-inf"))
        k = (mask.sum(1).float() * topk_ratio).floor().clamp_min(1).long()
        outs = []
        for b in range(B):
            if not mask[b].any():
                outs.append(torch.tensor(0., device=logits.device, dtype=logits.dtype))
            else:
                vals = torch.topk(x[b], k=int(min(k[b].item(), T))).values
                vals = vals[torch.isfinite(vals)]
                outs.append(vals.mean() if vals.numel() else torch.tensor(0., device=logits.device, dtype=logits.dtype))
        return torch.stack(outs, 0)
    else:  # "lse"
        x = logits.masked_fill(~mask, float("-inf"))
        out = torch.logsumexp(tau * x, dim=1) / tau
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

# ---------- 체크포인트 로드 ----------
def load_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # train.py 저장 형식 호환 :contentReference[oaicite:6]{index=6}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing)
    if unexpected:
        print("[warn] unexpected keys:", unexpected)
    return ckpt

@torch.no_grad()
def evaluate(model, loader, device, pool, tau, topk_ratio, threshold, out_csv=None, dataset_items=None):
    model.eval(); model.to(device)
    probs_all, preds_all, labels_all = [], [], []
    names = []

    seen = 0  # DataLoader는 shuffle=False 가정
    for batch in tqdm(loader, desc="Testing"):
        video = batch["video"].to(device, non_blocking=True)   # (B,T,N,3,H,W) :contentReference[oaicite:7]{index=7}
        audio = batch["audio"].to(device, non_blocking=True)   # (B,T,M,L)   :contentReference[oaicite:8]{index=8}
        mask  = batch["mask"].to(device, non_blocking=True)    # (B,T)       :contentReference[oaicite:9]{index=9}
        label = batch["label"].to(device, non_blocking=True)   # (B,)        :contentReference[oaicite:10]{index=10}

        logits = model(video, audio, mask)                     # (B,T)       :contentReference[oaicite:11]{index=11}
        bag_logit = apply_mil_pool(logits, mask, pool=pool, tau=tau, topk_ratio=topk_ratio)
        bag_prob  = torch.sigmoid(bag_logit)

        preds = (bag_prob >= threshold).long()

        probs_all.append(bag_prob.cpu().numpy())
        preds_all.append(preds.cpu().numpy())
        labels_all.append(label.cpu().numpy())

        # 샘플 이름(비디오 경로) 기록: dataset.items 순서를 활용
        if dataset_items is not None:
            B = label.shape[0]
            for i in range(B):
                idx = seen + i
                if 0 <= idx < len(dataset_items):
                    names.append(str(dataset_items[idx].get("video", f"idx{idx}")))
                else:
                    names.append(f"idx{idx}")
            seen += B

    probs_all = np.concatenate(probs_all, 0)
    preds_all = np.concatenate(preds_all, 0).astype(int)
    labels_all = np.concatenate(labels_all, 0).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels_all, preds_all)),
        "f1": float(f1_score(labels_all, preds_all)),
    }
    # AUC은 두 클래스가 모두 있을 때만
    if len(np.unique(labels_all)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels_all, probs_all))
    cm = confusion_matrix(labels_all, preds_all)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["report"] = classification_report(labels_all, preds_all, output_dict=True)

    # 저장
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # per-sample
        import csv
        with open(out_csv.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "label", "prob_pos", "pred"])
            for i in range(len(labels_all)):
                name = names[i] if i < len(names) else f"idx{i}"
                w.writerow([name, int(labels_all[i]), float(probs_all[i]), int(preds_all[i])])
        # summary
        with open(out_csv.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="설정 YAML 경로")  # csv_manifest, 모델 차원 등 :contentReference[oaicite:12]{index=12}
    ap.add_argument("--ckpt", required=True, help="학습된 체크포인트(.pt)")
    ap.add_argument("--csv_manifest", default=None, help="테스트 manifest(jsonl) 경로(미지정 시 config.yaml의 csv_manifest 사용)")
    ap.add_argument("--batch_size", type=int, default=None, help="없으면 config의 batch_size 사용")
    ap.add_argument("--num_workers", type=int, default=None, help="없으면 config의 num_workers 사용")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="runs/test/predictions")  # .csv / .metrics.json 로 저장
    # (선택) MIL 풀링 강제 오버라이드
    ap.add_argument("--mil_pool", choices=["mean","max","topk","lse"], default=None)
    ap.add_argument("--mil_tau", type=float, default=None)
    ap.add_argument("--mil_topk_ratio", type=float, default=None)
    args = ap.parse_args()

    # 1) 설정 로드
    cfg = load_config(args.config)  # dataclass Cfg 로드 :contentReference[oaicite:13]{index=13}
    if args.csv_manifest:
        cfg.csv_manifest = args.csv_manifest
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    # 2) 데이터
    ds = MultiModalMILDataset(cfg)                          # jsonl: {"video","audio","label"} 기대 :contentReference[oaicite:14]{index=14}
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=("cuda" in args.device)
    )

    # 3) 모델 만들고 가중치 로드
    model = MultiModalMILNet(
        n_mels=cfg.n_mels,
        dv=cfg.video_out_dim, da=cfg.audio_out_dim, df=cfg.fuse_dim
    )                                                       # 모델/forward 형식 :contentReference[oaicite:15]{index=15}
    ckpt = load_ckpt(model, args.ckpt)                      # train.py 저장 포맷 호환 로딩 :contentReference[oaicite:16]{index=16}

    # 4) MIL 풀링 옵션: config 우선, CLI로 오버라이드 가능
    pool = args.mil_pool if args.mil_pool else cfg.mil_pool
    tau  = args.mil_tau if args.mil_tau is not None else cfg.mil_tau
    topk = args.mil_topk_ratio if args.mil_topk_ratio is not None else cfg.mil_topk_ratio

    # 5) 평가
    metrics = evaluate(
        model, dl, args.device,
        pool=pool, tau=tau, topk_ratio=topk,
        threshold=args.threshold,
        out_csv=args.out,
        dataset_items=getattr(ds, "items", None)
    )

    print("\n== Test metrics ==")
    for k, v in metrics.items():
        if k not in ("report", "confusion_matrix"):
            print(f"{k}: {v}")
    print("confusion_matrix:", metrics["confusion_matrix"])

if __name__ == "__main__":
    main()
