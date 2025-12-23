from __future__ import annotations
import argparse, os, csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

# 그래프 저장용 (없으면 CSV만 저장)
try:
    import matplotlib
    matplotlib.use("Agg")  # GUI 없이 파일로 저장
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from mmil.config import load_config
from mmil.data import MultiModalMILDataset, collate_fn
from mmil.model import MultiModalMILNet
# 변경: mil_bce_loss → compute_loss
from mmil.losses import compute_loss

def save_loss_plot(losses, out_png: str):
    if plt is None:
        print("[WARN] matplotlib not available; skip plotting.")
        return
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 로그 파일 경로
    loss_csv_epoch = os.path.join(cfg.save_dir, "loss_epoch.csv")
    loss_png_epoch = os.path.join(cfg.save_dir, "loss_epoch.png")
    loss_csv_iter  = os.path.join(cfg.save_dir, "loss_iter.csv")

    # CSV 헤더 초기화
    with open(loss_csv_epoch, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "avg_loss"])
    with open(loss_csv_iter, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "iter_in_epoch", "global_step", "loss"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset & loader
    ds = MultiModalMILDataset(cfg)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    # Model
    model = MultiModalMILNet(
        n_mels=cfg.n_mels, dv=cfg.video_out_dim, da=cfg.audio_out_dim, df=cfg.fuse_dim
    ).to(device)

    # Optim
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    epoch_losses = []
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}")
        epoch_loss = 0.0
        n_steps = 0

        for it, batch in enumerate(pbar, start=1):
            video = batch["video"].to(device, non_blocking=True)   # (B,T,N,3,H,W)
            audio = batch["audio"].to(device, non_blocking=True)   # (B,T,M,L)
            mask  = batch["mask"].to(device, non_blocking=True)    # (B,T)
            label = batch["label"].to(device, non_blocking=True)   # (B,)

            logits = model(video, audio, mask)                     # (B,T)

            # 변경: 손실 계산
            loss = compute_loss(cfg, logits, label, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # 권장: 그래디언트 클리핑(스파이크 억제)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # 로깅
            global_step += 1
            n_steps += 1
            lval = float(loss.item())
            epoch_loss += lval
            pbar.set_postfix(loss=f"{lval:.4f}")

            with open(loss_csv_iter, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([epoch, it, global_step, lval])

        avg = epoch_loss / max(n_steps, 1)
        epoch_losses.append(avg)
        print(f"[Epoch {epoch}] avg loss = {avg:.4f}")

        with open(loss_csv_epoch, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, avg])

        save_loss_plot(epoch_losses, loss_png_epoch)

        ckpt_path = Path(cfg.save_dir) / f"mm_mil_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "cfg": vars(cfg)}, ckpt_path)
        print(f"Saved: {ckpt_path}")

if __name__ == "__main__":
    main()
