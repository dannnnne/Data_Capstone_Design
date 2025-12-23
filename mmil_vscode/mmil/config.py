# mmil/config.py
from dataclasses import dataclass, fields
from typing import Optional
import yaml

@dataclass
class Cfg:
    # paths
    csv_manifest: str
    save_dir: str

    # data
    clip_len_sec: float = 0.5
    vframes_per_clip: int = 8
    img_size: int = 224
    sr: int = 16000
    n_mels: int = 128
    mel_win_ms: int = 25
    mel_hop_ms: int = 10

    # train
    batch_size: int = 2
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0

    # model dims
    video_out_dim: int = 256
    audio_out_dim: int = 256
    fuse_dim: int = 256

    # loss weights (composite에서 사용)
    lam_mil: float = 1.0
    lam_sp: float = 0.0
    lam_norm: float = 0.0
    lam_tv: float = 0.0
    lam_ag: float = 0.0

    # MIL pooling (BCE-MIL에서 사용)
    mil_pool: str = "lse"      # lse | mean | max | topk
    mil_tau: float = 5.0
    mil_topk_ratio: float = 0.1

    # external binaries (optional)
    ffmpeg_bin: Optional[str] = None

    # === 새 필드(중요) ===
    loss_type: str = "bce_mil"          # 'bce_mil' | 'composite'
    margin: float = 0.0                 # composite mean-MIL용
    pos_weight: Optional[float] = None  # 클래스 불균형 보정(선택)

def _to_float(d, keys):
    for k in keys:
        if k in d and isinstance(d[k], str):
            try:
                d[k] = float(d[k])
            except Exception:
                pass

def _to_int(d, keys):
    for k in keys:
        if k in d and isinstance(d[k], str):
            try:
                d[k] = int(d[k])
            except Exception:
                pass

def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # dataclass에 정의된 키만 통과(모르는 키 무시)
    allowed = {f.name for f in fields(Cfg)}
    d = {k: v for k, v in raw.items() if k in allowed}
    unknown = [k for k in raw.keys() if k not in allowed]
    if unknown:
        print(f"[WARN] unknown config keys ignored: {unknown}")

    # 안전 캐스팅(따옴표로 들어온 숫자 방지)
    _to_float(d, [
        "clip_len_sec","lr","weight_decay","lam_mil","lam_sp",
        "lam_norm","lam_tv","lam_ag","mil_tau","mil_topk_ratio",
        "margin","pos_weight"  # ← 새로 추가
    ])
    _to_int(d, [
        "batch_size","epochs","num_workers","vframes_per_clip","img_size",
        "sr","n_mels","mel_win_ms","mel_hop_ms","video_out_dim","audio_out_dim","fuse_dim"
    ])

    return Cfg(**d)
