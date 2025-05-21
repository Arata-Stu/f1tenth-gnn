import lidar_graph
from torch_geometric.data import Data, Batch
import torch

def build_batch_graph(scan_data_batch, k_neighbors=3, edge_threshold=1.0, use_knn=True):
    """
    C実装されたLidarGraphを用いて、LiDARデータをバッチ処理でグラフに変換。
    
    Args:
        scan_data_batch (Tensor): [batch_size, num_points] のLiDARスキャンデータ
        k_neighbors (int): 各ノードに対して接続する近傍ノード数（use_knn=Trueのとき）
        edge_threshold (float): エッジとして接続可能な最大距離
        use_knn (bool): True なら k近傍探索, False なら 全探索（しきい値以下全接続）

    Returns:
        Batch: torch_geometric.data.Batch オブジェクト（各バッチのグラフ）
    """
    batch_size, num_points = scan_data_batch.size()
    
    # Cライブラリの初期化（k, threshold, use_knn を追加）
    lidar_graph.initialize(num_points, batch_size, k_neighbors, edge_threshold, int(use_knn))

    # Pythonリストに変換
    lidar_data = [scan_data_batch[i].tolist() for i in range(batch_size)]
    
    # C言語のバッチ処理に渡してグラフを生成
    edge_lists = lidar_graph.build_graph(lidar_data)

    # 各バッチのノード座標を取得
    node_positions_batch = lidar_graph.get_node_positions()

    graphs = []

    # 各バッチのノードとエッジを Data に変換
    for i, edge_list in enumerate(edge_lists):
        node_positions = node_positions_batch[i]
        x = torch.tensor(node_positions, dtype=torch.float32)
        
        if len(edge_list) > 0:
            edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([e[2] for e in edge_list], dtype=torch.float32).view(-1, 1)
        else:
            # エッジが無い場合でも Data が生成できるように空 tensor を用意
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)

    # PyG の Batch オブジェクトとしてまとめて返す
    return Batch.from_data_list(graphs)
