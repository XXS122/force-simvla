import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_occluded_comparison(file_path, output_dir="experiment_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(file_path, "r") as f:
        # 获取第一条轨迹
        obs = f['data/demo_0/obs']
        agent_rgb = obs['agentview_rgb']
        wrist_rgb = obs['eye_in_hand_rgb']
        
        # 选取几个关键时间步 (比如开始、中期、接近物体)
        indices = [0, 20, 50, 80]
        
        for idx in indices:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 获取原始图像
            raw_img = agent_rgb[idx].copy()
            
            # --- 模拟遮挡：在中心区域加黑块 ---
            # 假设物体在中心附近，遮挡 1/4 区域
            h, w, _ = raw_img.shape
            raw_img[h//4:3*h//4, w//4:3*w//4, :] = 0 
            
            # 显示结果
            axes[0].imshow(raw_img)
            axes[0].set_title(f"Occluded Agentview (Step {idx})")
            
            axes[1].imshow(wrist_rgb[idx])
            axes[1].set_title(f"Clear Wristview (Step {idx})")
            
            plt.savefig(f"{output_dir}/comparison_step_{idx}.png")
            plt.close()
            print(f"已生成对比图: {output_dir}/comparison_step_{idx}.png")

# 在你的 A100 上运行
generate_occluded_comparison("/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5")