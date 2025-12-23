# tools/batch_log.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # 리포 루트 디렉토리 기준으로 경로 잡기
    root_dir = Path(__file__).resolve().parents[1]      # 프로젝트 루트
    csv_path = root_dir / "checkpoints" / "loss_iter.csv"   # 파일 이름 다르면 여기만 수정

    if not csv_path.exists():
        raise FileNotFoundError(f"로그 파일을 찾을 수 없음: {csv_path}")

    # CSV 읽기 (epoch,iter_in_epoch,global_step,loss)
    df = pd.read_csv(csv_path)

    # 기본 그래프: global_step vs loss
    plt.figure(figsize=(10, 4))
    plt.plot(df["global_step"], df["loss"])
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training Loss per Batch")
    plt.grid(True)
    plt.tight_layout()

    # PNG로 저장
    out_path = csv_path.with_name("loss_iter.png")
    plt.savefig(out_path, dpi=200)
    print(f"saved plot to: {out_path}")

    # 필요하면 주석 해제해서 화면에도 띄울 수 있음
    # plt.show()


if __name__ == "__main__":
    main()
