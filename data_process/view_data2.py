import h5py
import numpy as np

def debug_libero_format(file_path):
    with h5py.File(file_path, "r") as f:
        # 1. 检查轨迹
        demos = list(f['data'].keys())
        print(f"总轨迹数: {len(demos)}")
        
        # 2. 读取第一条轨迹的观测项
        demo_key = demos[0]
        obs = f[f'data/{demo_key}/obs']
        actions = f[f'data/{demo_key}/actions']
        
        print(f"\n--- 轨迹 {demo_key} 详情 ---")
        print(f"任务指令: {f['data'].attrs.get('language_instruction', '无')}")
        
        # 3. 检查图像数据
        for img_key in ['agentview_rgb', 'eye_in_hand_rgb']:
            if img_key in obs:
                data = obs[img_key]
                print(f"[{img_key}]")
                print(f"  - 维度 (Shape): {data.shape}  (预期: [T, H, W, 3])")
                print(f"  - 类型 (Dtype): {data.dtype}")
                print(f"  - 数值范围: {np.min(data[0])} 到 {np.max(data[0])}")
        
        # 4. 检查动作数据
        print(f"\n[actions]")
        print(f"  - 维度 (Shape): {actions.shape}  (预期: [T, 7])")
        print(f"  - 类型 (Dtype): {actions.dtype}")
        
        # 5. 检查本体感受状态 (Robot States)
        if 'ee_states' in obs:
            print(f"\n[ee_states] (末端位姿)")
            print(f"  - 维度: {obs['ee_states'].shape}")

# 替换为你的 A100 服务器上的实际路径
debug_libero_format("/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5")