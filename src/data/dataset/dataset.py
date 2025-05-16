import os
import torch
from torch.utils.data import Dataset
import numpy as np
from src.utils.helper import linear_map

class ConcatLidarDataset(Dataset):
    def __init__(self, directory, target_points=60, max_steer=1.0, min_speed=0.0, max_speed=10.0):
        """
        指定されたディレクトリ内の全てのNpyファイルを結合して一つのデータセットとして扱います。

        Args:
            directory (str): Npyファイルが格納されたディレクトリ
            max_steer (float): ステアリングの最大値
            min_speed (float): スピードの最小値
            max_speed (float): スピードの最大値
        """
        self.files = []
        self.index_map = []
        self.max_steer = max_steer
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.target_points = target_points

        # ディレクトリ内の全てのnpyファイルを再帰的に探索
        for root, _, filenames in os.walk(directory):
            runs = set()
            for filename in sorted(filenames):
                if filename.startswith('run') and filename.endswith('_positions.npy'):
                    run_id = filename.split('_')[0]  # run0, run1, etc.
                    runs.add(run_id)
            
            # 各エピソードのデータをロードして長さを計算
            for run_id in sorted(runs):
                pos_path = os.path.join(root, f"{run_id}_positions.npy")
                positions = np.load(pos_path, mmap_mode='r')
                length = len(positions)
                self.files.append((root, run_id))
                self.index_map.extend([(root, run_id, idx) for idx in range(length)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        root, run_id, internal_idx = self.index_map[idx]

        # 各データを個別にロード (mmap_mode='r'で高速化)
        positions = np.load(os.path.join(root, f"{run_id}_positions.npy"), mmap_mode='r')[internal_idx].copy()
        scans = np.load(os.path.join(root, f"{run_id}_scans.npy"), mmap_mode='r')[internal_idx].copy()
        waypoints = np.load(os.path.join(root, f"{run_id}_waypoints.npy"), mmap_mode='r')[internal_idx].copy()
        prev_actions = np.load(os.path.join(root, f"{run_id}_prev_actions.npy"), mmap_mode='r')[internal_idx].copy()
        actions = np.load(os.path.join(root, f"{run_id}_actions.npy"), mmap_mode='r')[internal_idx].copy()

        # --- Scansの次元調整 ---
        if len(scans.shape) == 2 and scans.shape[0] == 1:
            scans = scans.squeeze(0)

        # --- 1. Waypointsの座標を自己位置基準に変換 + 速度の正規化 ---
        if len(waypoints.shape) == 3 and waypoints.shape[0] == 1:
            waypoints = waypoints.squeeze(0)

        # **ここが重要**: copy() を使って書き込み可能な配列に変換
        waypoints[:, :2] -= positions[:2]

        # 速度 (2) は線形正規化
        waypoints[:, 2] = linear_map(waypoints[:, 2], self.min_speed, self.max_speed, 0, 1)

        # --- 2. Scansのダウンサンプル ---
        original_indices = np.linspace(0, len(scans) - 1, num=len(scans))
        target_indices = np.linspace(0, len(scans) - 1, num=self.target_points)
        scans = np.interp(target_indices, original_indices, scans.flatten())
        scans = scans.reshape(self.target_points)

        # --- 3. Actionsの正規化 ---
        if len(actions.shape) == 3 and actions.shape[0] == 1:
            actions = actions.squeeze(0)
        elif len(actions.shape) == 2 and actions.shape[0] == 1:
            actions = actions.squeeze(0)
        elif len(actions.shape) == 1 and actions.shape[0] == 1:
            actions = np.pad(actions, (0, 1), mode='constant')

        # 正規化
        actions[0] /= self.max_steer
        actions[1] = linear_map(actions[1], self.min_speed, self.max_speed, 0, 1)

        # --- 最後にTensorに変換 ---
        item = {
            'positions': torch.tensor(positions, dtype=torch.float32),
            'scans': torch.tensor(scans, dtype=torch.float32),
            'waypoints': torch.tensor(waypoints, dtype=torch.float32),
            'prev_actions': torch.tensor(prev_actions, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.float32)
        }

        return item

