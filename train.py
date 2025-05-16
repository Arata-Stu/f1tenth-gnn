import os
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset.dataset import ConcatLidarDataset
from src.models.model import build_model
from src.data.graph.graph import build_batch_graph
from src.utils.timers import Timer

class EarlyStopping:
    """訓練損失を監視し、patience エポック改善がなければ学習停止"""
    def __init__(self, patience=5, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.verbose:
                print(f"  ➤ Improvement detected (loss: {current_loss:.4f}), reset counter.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ➤ No improvement ({current_loss:.4f}), counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    # 設定表示
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データローダー
    dataset = ConcatLidarDataset(
        directory=cfg.dataset.path,
        target_points=cfg.dataset.target_points,
        max_steer=cfg.dataset.max_steer,
        min_speed=cfg.dataset.min_speed,
        max_speed=cfg.dataset.max_speed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )
    print(f"Dataset size: {len(dataset)}")

    # モデル／最適化設定
    model = build_model(model_cfg=cfg.model).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)

    # EarlyStopping インスタンス
    early_stopper = EarlyStopping(
        patience=cfg.training.early_stop_patience,
        verbose=True
    )

    # チェックポイント保存先
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        cfg.training.checkpoint_dir,
        cfg.training.checkpoint_filename
    )

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(dataloader, start=1):
            scans = batch_data['scans'].to(device)
            actions_true = batch_data['actions'].to(device)

            with Timer("Batch Graph Creation"):
                graph_batch = build_batch_graph(scans).to(device)

            optimizer.zero_grad()
            with Timer("Model Forward Pass"):
                actions_pred = model(graph_batch)

            loss = criterion(actions_pred, actions_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % cfg.training.log_interval == 0:
                avg_loss = total_loss / batch_idx
                print(f"Epoch[{epoch}/{cfg.num_epochs}] "
                      f"Batch[{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (Avg: {avg_loss:.4f})")

        # エポック終了後の平均損失
        epoch_loss = total_loss / len(dataloader)
        print(f"====> Epoch {epoch} Average loss: {epoch_loss:.4f}")

        # EarlyStopping 判定
        early_stopper(epoch_loss)
        # 最良モデルの保存
        if epoch_loss <= early_stopper.best_loss + 1e-8:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✅ Model saved to {checkpoint_path}")

        if early_stopper.early_stop:
            print(f"☆ Early stopping triggered (no improvement for {cfg.training.early_stop_patience} epochs).")
            break

    print("Training finished.")

if __name__ == "__main__":
    main()
