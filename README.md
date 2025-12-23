# Multimodal MIL (Video + Audio) — VS Code Starter

This is a **minimal, self-contained** PyTorch project to train a bag-level classifier with **Multiple Instance Learning (MIL)** using **online preprocessing**:
- **Video** → per-clip frame sampling (OpenCV), resized to 224×224
- **Audio** → Mel-spectrogram (librosa), sliced per clip
- **Fusion** → Gate-based fusion (audio/video)
- **Temporal Head** → ED-TCN-like 1D conv head
- **Loss** → MIL (BCE + LSE pooling by default)

> Built to mirror the notebook you were using, but runnable as plain Python from VS Code.

---

## 1) Create & activate env (Windows PowerShell)

```powershell
# Choose one:
# (A) conda
conda create -n mmil python=3.10 -y
conda activate mmil

# (B) venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

> **Install PyTorch first** (CUDA build that matches your GPU).
> Follow https://pytorch.org/get-started/locally/ then return here.

```powershell
# Example (CUDA 12.4) — check the official website for the exact command for your GPU
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Then install the rest:
pip install -r requirements.txt
```

If OpenCV cannot decode some videos, install ffmpeg on your system and/or try a different OpenCV build.

## 3) Prepare a manifest (JSONL)

Create a file like `data/train_manifest.jsonl` (one JSON per line):

```json
{"video": "C:/data/fall01.mp4", "audio": "C:/data/fall01.wav", "label": 1}
{"video": "C:/data/normal01.mp4", "audio": null,             "label": 0}
```

> `audio` can be `null`. The loader will replace it with silence for that sample.

## 4) Edit `config.yaml`

- `csv_manifest` → your JSONL path
- `save_dir` → where to store checkpoints
- `clip_len_sec`, `vframes_per_clip`, `sr`, `n_mels` as needed

## 5) Run training

```powershell
python train.py --config config.yaml
```

- On Windows, the code uses `num_workers=0` by default.
- Checkpoints will be written to `checkpoints/`.

## 6) Notes

- This starter uses **lightweight stubs** for video/audio encoders so you can test the wiring quickly.
- Later, swap in real backbones (VideoMAE, PANNs) inside `mmil/model.py`.
- To start with **MIL only**, it's already configured (other losses are set to 0).

---

## Project layout

```
mmil_vscode/
├─ config.yaml
├─ requirements.txt
├─ train.py
├─ README.md
└─ mmil/
   ├─ __init__.py
   ├─ config.py
   ├─ data.py
   ├─ losses.py
   └─ model.py
```

Happy training!
