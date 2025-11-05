from __future__ import annotations
import os, json, subprocess, hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import librosa

# -------------------- JSONL utils --------------------

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def _uniform_sample_indices(start: int, end: int, k: int) -> List[int]:
    # inclusive start, exclusive end
    total = max(end - start, 1)
    if k <= 1:
        return [start]
    step = total / k
    return [int(start + i * step) for i in range(k)]

# -------------------- Audio helpers --------------------

def _extract_audio_wav_cached(video_path: str, sr: int, cache_dir: str = ".cache_audio", ffmpeg_bin: str = "ffmpeg") -> str:
    """
    FFmpeg CLI로 영상에서 오디오를 추출해 캐시에 wav로 저장하고 경로 반환.
    동일 파일(수정시간)·샘플레이트 조합은 재사용.
    """
    Path(cache_dir).mkdir(exist_ok=True)
    try:
        mtime = os.path.getmtime(video_path)
    except Exception:
        mtime = 0
    key = f"{video_path}|{sr}|{mtime}".encode("utf-8", "ignore")
    name = hashlib.md5(key).hexdigest() + f"_{sr}.wav"
    out = str(Path(cache_dir) / name)

    if not os.path.exists(out):
        cmd = [ffmpeg_bin, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", out]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out


# -------------------- Dataset --------------------

class MultiModalMILDataset(Dataset):
    """
    Online preprocessing:
      - Video: clip 나눠 프레임 샘플링/리사이즈 → (T, N, 3, H, W)
      - Audio: MP4 내 오디오(or 별도 wav) → 멜스펙 계산 후 clip 단위 슬라이스 → (T, M, Lclip)
      - Label: bag 레이블(0/1)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.items = _read_jsonl(cfg.csv_manifest)

        # mel timing
        self.win_length = int(cfg.mel_win_ms / 1000 * cfg.sr)
        self.hop_length = int(cfg.mel_hop_ms / 1000 * cfg.sr)

    def __len__(self):
        return len(self.items)

    # ---------- Video ----------

    def _load_video(self, path: str) -> Tuple[np.ndarray, float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)  # BGR
        cap.release()
        frames = np.array(frames, dtype=np.uint8)  # (F, H, W, 3)
        if frames.size == 0:
            raise RuntimeError(f"No frames decoded from: {path}")
        if fps < 1e-6:
            fps = 30.0  # fallback
        return frames, float(fps)

    def _prepare_video_clips(self, frames: np.ndarray, fps: float) -> np.ndarray:
        cfg = self.cfg
        F = frames.shape[0]
        total_sec = F / fps
        T = max(int(total_sec / cfg.clip_len_sec), 1)
        clip_frames = int(round(cfg.clip_len_sec * fps))

        H = W = cfg.img_size
        out = np.zeros((T, cfg.vframes_per_clip, 3, H, W), dtype=np.float32)

        for t in range(T):
            start = t * clip_frames
            end = min((t + 1) * clip_frames, F)
            idxs = _uniform_sample_indices(start, end, cfg.vframes_per_clip)
            for i, idx in enumerate(idxs):
                idx = min(idx, F - 1)
                img = frames[idx]  # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # C,H,W
                out[t, i] = img
        return out  # (T, N, 3, H, W)

    # ---------- Audio (MP4 내장 오디오 지원) ----------

    def _load_audio_from_video_or_path(self, audio_path: str | None, video_path: str, target_len_sec: float) -> np.ndarray:
        cfg = self.cfg

        # 1) 별도 오디오 파일 경로가 있으면 그대로 로드
        if audio_path is not None:
            y, _ = librosa.load(audio_path, sr=cfg.sr, mono=True)
            return y

        # 2) torchaudio StreamReader로 mp4에서 직접 디코딩 시도
        try:
            import torchaudio
            s = torchaudio.io.StreamReader(video_path)
            s.add_audio_stream(frames_per_chunk=0, sample_rate=cfg.sr)
            chunks = []
            for (chunk,) in s.stream():
                # chunk: (channels, frames)
                chunks.append(chunk)
            if len(chunks) > 0:
                wav = torch.cat(chunks, dim=1).mean(dim=0)  # mono
                return wav.cpu().numpy().astype(np.float32)
        except Exception:
            pass  # 실패 시 다음 단계

        # 3) FFmpeg CLI로 캐시 wav 추출 후 로드
        try:
            wav_path = _extract_audio_wav_cached(
                video_path, sr=cfg.sr,
                ffmpeg_bin=(getattr(self.cfg, "ffmpeg_bin", None) or "ffmpeg"))
            y, _ = librosa.load(wav_path, sr=cfg.sr, mono=True)
            return y
        except Exception as e:
            print(f"[WARN] Could not extract audio from {video_path}: {e}. Using silence.")
            return np.zeros(int(target_len_sec * cfg.sr), dtype=np.float32)

    def _load_audio(self, path: str | None, target_len_sec: float, video_path_for_fallback: str | None = None) -> np.ndarray:
        """
        path가 None이면 video_path_for_fallback(mp4)에서 오디오를 추출 시도.
        """
        if path is None and video_path_for_fallback is not None:
            y = self._load_audio_from_video_or_path(None, video_path_for_fallback, target_len_sec)
        else:
            y, _ = librosa.load(path, sr=self.cfg.sr, mono=True)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.cfg.sr,
            n_fft=2048,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.cfg.n_mels,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db  # (M, Ltotal)

    # ---------- Slice mel per clip ----------

    def _slice_mel_by_clips(self, mel_db: np.ndarray, total_sec: float) -> np.ndarray:
        cfg = self.cfg
        t_per = self.hop_length / cfg.sr          # seconds per mel frame
        Ltotal = mel_db.shape[1]

        T = max(int(total_sec / cfg.clip_len_sec), 1)
        Lclip = max(int(round(cfg.clip_len_sec / t_per)), 1)

        out = np.zeros((T, cfg.n_mels, Lclip), dtype=np.float32)
        for t in range(T):
            start_t = int(round((t * cfg.clip_len_sec) / t_per))
            end_t = min(start_t + Lclip, Ltotal)
            seg = mel_db[:, start_t:end_t]
            if seg.shape[1] < Lclip:
                pad_val = seg.min() if seg.size > 0 else -80.0
                pad = np.full((cfg.n_mels, Lclip - seg.shape[1]), pad_val, dtype=np.float32)
                seg = np.concatenate([seg, pad], axis=1)
            out[t] = seg.astype(np.float32)
        return out  # (T, M, Lclip)

    # ---------- __getitem__ ----------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cfg = self.cfg
        it = self.items[idx]
        video_path = it["video"]
        audio_path = it.get("audio", None)
        label = int(it["label"])

        frames, fps = self._load_video(video_path)
        video_clips = self._prepare_video_clips(frames, fps)  # (T, N, 3, H, W)

        total_sec = frames.shape[0] / fps
        mel_db = self._load_audio(audio_path, total_sec, video_path_for_fallback=video_path)
        audio_clips = self._slice_mel_by_clips(mel_db, total_sec)  # (T, M, Lclip)

        T = video_clips.shape[0]
        mask = np.ones((T,), dtype=np.bool_)

        return {
            "video": torch.from_numpy(video_clips),  # (T, N, 3, H, W)
            "audio": torch.from_numpy(audio_clips),  # (T, M, Lclip)
            "mask": torch.from_numpy(mask),          # (T,)
            "label": torch.tensor(label, dtype=torch.long),
        }

# -------------------- Collate --------------------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # pad along T to T_max
    T_list = [b["mask"].shape[0] for b in batch]
    T_max = max(T_list)
    N = batch[0]["video"].shape[1]
    C, H, W = batch[0]["video"].shape[2:5]
    M = batch[0]["audio"].shape[1]
    Lclip = batch[0]["audio"].shape[2]

    B = len(batch)
    videos = torch.zeros((B, T_max, N, C, H, W), dtype=torch.float32)
    audios = torch.zeros((B, T_max, M, Lclip), dtype=torch.float32)
    masks  = torch.zeros((B, T_max), dtype=torch.bool)
    labels = torch.zeros((B,), dtype=torch.long)

    for i, b in enumerate(batch):
        T = b["mask"].shape[0]
        videos[i, :T] = b["video"]
        audios[i, :T] = b["audio"]
        masks[i, :T]  = b["mask"]
        labels[i] = b["label"]

    return {"video": videos, "audio": audios, "mask": masks, "label": labels}
