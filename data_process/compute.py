import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
import os

# 1. 离线模型加载
model_path = "/datasets/models/base_model"
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    local_files_only=True,
    trust_remote_code=True
).cuda().eval()

# 模拟动作投影层
torch.manual_seed(42)
projection_layer = torch.nn.Linear(7, 768).cuda().to(torch.bfloat16)

def get_emb(img_array):
    """提取视觉特征并确保输出为 [768] 的向量"""
    inputs = processor(images=img_array, return_tensors="pt").to("cuda", torch.bfloat16)
    
    # 处理 SmolVLM 的 5 维输入
    if "pixel_values" in inputs and inputs["pixel_values"].ndim == 5:
        b, n, c, h, w = inputs["pixel_values"].shape
        inputs["pixel_values"] = inputs["pixel_values"].view(b * n, c, h, w)
    
    if "patch_attention_mask" in inputs and inputs["patch_attention_mask"].ndim == 4:
        b, n, ph, pw = inputs["patch_attention_mask"].shape
        inputs["patch_attention_mask"] = inputs["patch_attention_mask"].view(b * n, ph, pw)

    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        # 显式 mean 和 flatten，确保返回的是平铺的特征向量
        return outputs.last_hidden_state.mean(dim=1).flatten() 

def analyze_correlation(file_path):
    with h5py.File(file_path, "r") as f:
        demo = f['data/demo_0']
        obs = demo['obs']
        # 确保动作类型正确
        actions = torch.from_numpy(demo['actions'][:]).cuda().to(torch.bfloat16)
        
        agent_rgb = obs['agentview_rgb'][:]
        wrist_rgb = obs['eye_in_hand_rgb'][:]
        
        num_frames = 50
        results = {"raw": [], "occ": [], "wrist": []}

        print(f"正在分析 A100 数据...")
        for i in range(num_frames):
            # 视觉处理
            img_raw = agent_rgb[i]
            img_occ = img_raw.copy()
            img_occ[32:96, 32:96, :] = 0 
            img_wrist = wrist_rgb[i]

            # 提取特征
            e_raw = get_emb(img_raw)
            e_occ = get_emb(img_occ)
            e_wrist = get_emb(img_wrist)

            # 动作投影
            with torch.no_grad():
                a_emb = projection_layer(actions[i]).flatten()

            # --- 核心修复：防错相似度计算 ---
            def safe_calc_sim(vec_e, vec_a):
                # 强制转化为 [1, 768] 的标准形状进行对比
                v1 = vec_e.reshape(1, -1)
                v2 = vec_a.reshape(1, -1)
                return F.cosine_similarity(v1, v2, dim=1).item()

            results["raw"].append(safe_calc_sim(e_raw, a_emb))
            results["occ"].append(safe_calc_sim(e_occ, a_emb))
            results["wrist"].append(safe_calc_sim(e_wrist, a_emb))
            
            if i % 10 == 0:
                print(f"进度: {i}/{num_frames} | Raw Sim: {results['raw'][-1]:.4f}")

        return results

def plot_results(stats):
    labels = ['Raw Agentview', 'Occluded Agentview', 'Wristview']
    means = [np.mean(stats["raw"]), np.mean(stats["occ"]), np.mean(stats["wrist"])]
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = plt.bar(labels, means, color=colors, edgecolor='black', width=0.5)
    
    plt.ylabel('Cosine Similarity (Feature vs Action)')
    plt.title('Correlation Analysis: Why We Need Adaptive Fusion')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.4f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.savefig("correlation_results.png", dpi=300)
    print("柱状图已生成: correlation_results.png")

if __name__ == "__main__":
    data_path = "/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"
    if os.path.exists(data_path):
        stats = analyze_correlation(data_path)
        plot_results(stats)