import numpy as np
import torch
import hydra

from omegaconf import DictConfig, OmegaConf
from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from src.envs.envs import make_env
from src.planner.purePursuit import PurePursuitPlanner
from src.utils.timers import Timer
import lidar_graph

import matplotlib.pyplot as plt
import signal
import sys

from torch_geometric.data import Data
from src.models.gnn import N2LidarGCN, N4LidarGCN

# --- 可視化モジュール ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

def signal_handler(sig, frame):
    print("Gracefully shutting down...")
    plt.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
    
def prepare_graph_for_gcn(nodes, edges):
    """
    LiDARのノードとエッジの情報をGCN用のData形式に変換する
    """
    # ノードの位置を特徴量としてTensor化
    nodes_arr = np.array(nodes, dtype=np.float32)
    x = torch.tensor(nodes_arr, dtype=torch.float32)
    
    # エッジのインデックスを抽出
    edge_index = np.array([[src, dst] for src, dst, _ in edges], dtype=np.int64).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Dataオブジェクトの生成
    data = Data(x=x, edge_index=edge_index)
    return data

def visualize_graph(nodes, edges):
    ax.clear()
    ax.set_title("Lidar Sliding-Window Graph")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)

    nodes_arr = np.array(nodes)
    n_nodes = len(nodes_arr)

    # カラーマップの生成（インデックスに応じて色を変える）
    colors = plt.cm.jet(np.linspace(0, 1, n_nodes))

    # ノードの描画 (インデックスに基づく色分け)
    for i, (x, y) in enumerate(nodes_arr):
        ax.scatter(x, y, color=colors[i], s=5)

    # エッジの描画
    for src, dst, w in edges:
        x0, y0 = nodes_arr[src]
        x1, y1 = nodes_arr[dst]
        
        # エッジも同じようにグラデーションで表示
        color = colors[src]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=0.3)

    plt.pause(0.01)

@hydra.main(config_path="config", config_name="sample", version_base="1.2")
def main(cfg: DictConfig):
    # 設定表示
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # 環境・プランナ初期化
    map_manager = MapManager(
        map_name=cfg.envs.map.name,
        map_ext=cfg.envs.map.ext,
        speed=cfg.envs.map.speed,
        downsample=cfg.envs.map.downsample,
        use_dynamic_speed=cfg.envs.map.use_dynamic_speed,
        a_lat_max=cfg.envs.map.a_lat_max,
        smooth_sigma=cfg.envs.map.smooth_sigma
    )
    env = make_env(cfg.envs, map_manager, cfg.vehicle)
    planner = PurePursuitPlanner(
        wheelbase=cfg.planner.wheelbase,
        map_manager=map_manager,
        lookahead=cfg.planner.lookahead,
        gain=cfg.planner.gain,
        max_reacquire=cfg.planner.max_reacquire,
    )


    # 動的LiDARグラフ（スライディングウィンドウ2フレーム）
    lidar_graph.initialize(1080)
    gcn_model = N4LidarGCN(2, 256, 2, pool_method='mean')

    # 各レイヤーのパラメータ数を表示
    for name, param in gcn_model.named_parameters():
        print(f"{name}: {param.numel()}")

    # 総パラメータ数
    total_params = sum(p.numel() for p in gcn_model.parameters())
    print(f"Total number of parameters: {total_params}")

    while True:
        obs, info = env.reset()
        done = False
        timestamp = 0.0

        while not done:
            # 1) センサー読み取り
            scan = obs['scans'][0]              # numpy.ndarray
            scan_list = scan.tolist()              # list

            # 2) グラフ更新（タイム計測）
            with Timer("graph.forward"):
                edges = lidar_graph.build_graph(scan_list)
                nodes = lidar_graph.get_node_positions()

            data = prepare_graph_for_gcn(nodes, edges)

            # 4) GCNでのフォワードパス
            with Timer("gcn.forward"):
                output = gcn_model(data)
                print("GCN Output Shape:", output.shape)

            # 6) 可視化（タイム計測）
            # with Timer("visualize_graph"):
            #     visualize_graph(nodes, edges)

            # 7) プランナーで次アクション取得
            steer, speed = planner.plan(obs, id=0)
            action = [steer, speed]

            # 8) 環境ステップ
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated

            # 9) タイムスタンプ更新
            timestamp += 0.1

    # -- while end --

if __name__ == "__main__":
    main()
