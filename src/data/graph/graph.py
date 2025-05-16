import lidar_graph
from torch_geometric.data import Data, Batch
import torch

def build_batch_graph(scan_data_batch):
    """
    C実装されたLidarGraphを用いて、LiDARデータをバッチ処理でグラフに変換。
    
    Args:
        scan_data_batch (Tensor): [batch_size, num_points] のLiDARスキャンデータ
    Returns:
        Batch: torch_geometricのバッチデータ
    """
    batch_size, num_points = scan_data_batch.size()
    
    # Cライブラリの初期化
    lidar_graph.initialize(num_points, batch_size)

    # Pythonリストに変換
    lidar_data = [scan_data_batch[i].tolist() for i in range(batch_size)]
    
    # C言語のバッチ処理に渡してグラフを生成
    edge_lists = lidar_graph.build_graph(lidar_data)

    # **全てのバッチのノード座標を取得**
    node_positions_batch = lidar_graph.get_node_positions()

    graphs = []

    # **修正ポイント：各バッチのノードを個別に取り出す**
    for i, edge_list in enumerate(edge_lists):
        
        # 特定のバッチのノードだけを抜き出し
        node_positions = node_positions_batch[i]
        
        # 特徴量としてx, y座標を使用
        x = torch.tensor(node_positions, dtype=torch.float32)

        # エッジのリストを展開してテンソルに変換
        edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([e[2] for e in edge_list], dtype=torch.float).view(-1, 1)
        
        # PyGのDataオブジェクトを作成
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)

    # PyGのBatchデータを生成
    batch = Batch.from_data_list(graphs)
    
    return batch
