import os
import numpy as np
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager
from f1tenth_gym.maps.map_manager import TRAIN_MAP_DICT as MAP_DICT
from src.planner.purePursuit import PurePursuitPlanner

@hydra.main(config_path="config", config_name="collect_data", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # 実行ごとに固有のランIDディレクトリを作成
    base_out = cfg.output_dir
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(base_out, run_id)
    os.makedirs(out_root, exist_ok=True)

    # マップごとのディレクトリを事前に作成
    for map_name in MAP_DICT.values():
        os.makedirs(os.path.join(out_root, map_name), exist_ok=True)

    # 環境とプランナーの初期化
    map_cfg     = cfg.envs.map
    map_manager = MapManager(
        map_name=MAP_DICT[0],
        map_ext=map_cfg.ext,
        speed=map_cfg.speed,
        downsample=map_cfg.downsample,
        use_dynamic_speed=map_cfg.use_dynamic_speed,
        a_lat_max=map_cfg.a_lat_max,
        smooth_sigma=map_cfg.smooth_sigma
    )
    env = make_env(env_cfg=cfg.envs, map_manager=map_manager, param=cfg.vehicle)

    wheelbase = cfg.planner.wheelbase
    lookahead = cfg.planner.lookahead
    planner = PurePursuitPlanner(
        wheelbase=wheelbase,
        map_manager=map_manager,
        lookahead=lookahead,
        gain=cfg.planner.gain,
        max_reacquire=cfg.planner.max_reacquire,
    )

    render_flag = cfg.render
    render_mode = cfg.render_mode
    num_episodes  = cfg.num_episodes
    num_steps     = cfg.num_steps
    num_waypoints = cfg.get('num_waypoints', 10)

    map_counters = {m: 0 for m in MAP_DICT.values()}

    for ep in range(num_episodes):
        map_id = ep % len(MAP_DICT)
        name   = MAP_DICT[map_id]
        env.update_map(map_name=name, map_ext=map_cfg.ext)
        obs, info = env.reset()

        count = map_counters[name]
        map_counters[name] += 1

        # 保存先パスを定義
        map_dir = os.path.join(out_root, name)
        os.makedirs(map_dir, exist_ok=True)

        # --- データの初期化 ---
        positions = []
        scans = []
        waypoints = []
        prev_actions = []
        actions = []

        prev_action = np.zeros((1, 2), dtype='float32')
        current_pos = info.get('current_pos', np.array([0.0, 0.0], dtype='float32'))
        truncated = False

        for step in range(num_steps):
            steer, speed = planner.plan(obs)
            action = np.array([steer, speed], dtype='float32').reshape(1, 2)
            scan   = obs['scans'][0].astype('float32').reshape(1, cfg.envs.num_beams)

            wpts = map_manager.get_future_waypoints(
                current_pos, num_points=num_waypoints
            ).astype('float32')
            if wpts.shape[0] < num_waypoints:
                pad = np.repeat(wpts[-1][None, :], num_waypoints - wpts.shape[0], axis=0)
                wpts = np.vstack([wpts, pad])
            wpts = wpts.reshape(1, num_waypoints, 3)

            # --- データの保存 ---
            positions.append(current_pos)
            scans.append(scan)
            waypoints.append(wpts)
            prev_actions.append(prev_action)
            actions.append(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

            obs = next_obs
            prev_action = action
            current_pos = info.get('current_pos', current_pos)

            if render_flag:
                env.render(mode=render_mode) if render_mode else env.render()

        if not truncated:
            # --- データの書き込み ---
            np.save(os.path.join(map_dir, f"run{count}_positions.npy"), np.array(positions))
            np.save(os.path.join(map_dir, f"run{count}_scans.npy"), np.array(scans))
            np.save(os.path.join(map_dir, f"run{count}_waypoints.npy"), np.array(waypoints))
            np.save(os.path.join(map_dir, f"run{count}_prev_actions.npy"), np.array(prev_actions))
            np.save(os.path.join(map_dir, f"run{count}_actions.npy"), np.array(actions))
            
            print(f"Episode {ep} completed, saved to: {map_dir}")

if __name__ == '__main__':
    main()
