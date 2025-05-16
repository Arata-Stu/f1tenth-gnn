import numpy as np
from collections import deque
import torch


def linear_map(x, x_min, x_max, y_min, y_max):
    """Linear mapping function."""
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def inverse_linear_map(y, x_min, x_max, y_min, y_max):
    """
    線形マッピングの逆関数（denormalize）。
    y_min～y_max の範囲にマップされた値 y を、
    元の x_min～x_max の範囲に戻します。
    """
    return (y - y_min) / (y_max - y_min) * (x_max - x_min) + x_min

class ScanBuffer:
    def __init__(self, num_beams: int = 1080,
                 buffer_size: int = 2,
                 downsample_size: int = 60):
        """
        num_beams: 1フレームのLiDARスキャンの長さ
        buffer_size: 保持するスキャンフレームの数
        downsample_size: 各スキャンをこの長さにダウンサンプリングする
        """
        self.num_beams = num_beams
        self.buffer_size = buffer_size
        self.downsample_size = downsample_size
        self.scan_buffer = deque(maxlen=buffer_size)

    def add_scan(self, scan: np.ndarray):
        """新しいスキャンを追加する"""
        if scan.shape[0] != self.num_beams:
            raise ValueError(f"scan length {scan.shape[0]} != expected {self.num_beams}")
        self.scan_buffer.append(scan)

    def is_full(self) -> bool:
        """バッファが満タンかどうかを確認する"""
        return len(self.scan_buffer) == self.buffer_size
    
    def reset(self):
        """スキャンバッファをクリアする"""
        self.scan_buffer.clear()

    def _pad_frames(self, frames: list) -> list:
        """
        バッファが満たない場合、最後のフレームを繰り返して埋める
        """
        if not frames:
            raise ValueError("No frames in buffer")
        if len(frames) < self.buffer_size:
            last = frames[-1]
            frames = frames + [last] * (self.buffer_size - len(frames))
        return frames

    def _downsample(self, arr: np.ndarray) -> np.ndarray:
        """
        1次元の配列をdownsample_sizeにダウンサンプリングする
        """
        if self.downsample_size is None or arr.size == self.downsample_size:
            return arr
        indices = np.linspace(0, arr.size - 1, self.downsample_size, dtype=int)
        return arr[indices]

    def get_concatenated_numpy(self) -> np.ndarray:
        """
        結合されたフレームをNumPy配列として返す
        """
        frames = list(self.scan_buffer)
        frames = self._pad_frames(frames)
        processed = [self._downsample(f) for f in frames]
        return np.hstack(processed)

    def get_concatenated_tensor(self,
                                device: torch.device = None,
                                dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        結合されたフレームをPyTorchテンソルとして返す
        """
        frames = list(self.scan_buffer)
        frames = self._pad_frames(frames)
        tensors = []
        for f in frames:
            arr = self._downsample(f) if isinstance(f, np.ndarray) else f.numpy()
            t = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else f
            tensors.append(t)
        out = torch.cat(tensors, dim=0)
        if device:
            out = out.to(device)
        if dtype:
            out = out.to(dtype)
        return out