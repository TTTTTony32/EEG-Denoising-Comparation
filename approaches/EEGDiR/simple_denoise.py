"""
简化版降噪脚本
用于在测试集上进行降噪并输出完整的(2700,512)形状的numpy矩阵
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AbstractDenoiser
from utils.train_valid_utils import get_config, init_model, load_dataset
from accelerate import Accelerator


def simple_denoise():
    """简化的降噪函数"""
    
    # ==================== 配置参数 ====================
    config_path = "config/retnet/config.yml"
    weight_path = "./results/2024_01_11_15_DiR_4_EOG_pathch16_mini_seq32_hidden_dim512_layer_1_EMG/weight/best.pth"
    output_path = "./denoised_test_output.npy"
    
    # ==================== 检查文件 ====================
    if not os.path.exists(weight_path):
        print(f"错误: 权重文件不存在: {weight_path}")
        print("请先训练模型或检查权重文件路径")
        return None
    
    if not os.path.exists("./datasets/test_input.npy"):
        print("错误: 测试数据文件不存在")
        print("请确保 ./datasets/test_input.npy 和 ./datasets/test_output.npy 文件存在")
        return None
    
    # ==================== 加载模型和数据 ====================
    print("1. 加载配置文件...")
    config = get_config(config_path)
    config["model"]["weight_path"] = weight_path
    
    print("2. 初始化模型...")
    model = init_model(config)
    
    print("3. 加载测试数据...")
    test_x = np.load("./datasets/test_input.npy")
    test_y = np.load("./datasets/test_output.npy")
    print(f"   测试数据形状: {test_x.shape}")
    
    # ==================== 准备数据加载器 ====================
    test_data = {
        'train_x': test_x,
        'train_y': test_y,
        'test_x': test_x,
        'test_y': test_y,
    }
    _, test_dataset = load_dataset(config, custom_data=test_data)
    
    batch_size = config["train"]["batch_size"]
    data_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    # ==================== 设置设备 ====================
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    
    # ==================== 执行降噪 ====================
    print("4. 开始降噪处理...")
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="降噪进度"):
            inputs = batch["y"]
            outputs, _ = model(inputs, inputs)  # 使用inputs作为labels（推理时不需要真实标签）
            
            outputs = accelerator.gather_for_metrics(outputs)
            outputs_np = outputs.cpu().detach().numpy()
            all_predictions.append(outputs_np)
    
    # ==================== 合并结果 ====================
    print("5. 合并结果...")
    denoised_output = np.concatenate(all_predictions, axis=0)
    
    # 确保输出形状正确
    if denoised_output.shape[0] > test_x.shape[0]:
        denoised_output = denoised_output[:test_x.shape[0], :]
    
    print(f"   输出形状: {denoised_output.shape}")
    
    # ==================== 保存结果 ====================
    print("6. 保存结果...")
    np.save(output_path, denoised_output)
    
    # 同时保存为txt格式
    txt_path = output_path.replace('.npy', '.txt')
    np.savetxt(txt_path, denoised_output, fmt='%.6f')
    
    print(f"   已保存到: {output_path}")
    print(f"   已保存到: {txt_path}")
    
    return denoised_output


if __name__ == "__main__":
    result = simple_denoise()
    if result is not None:
        print(f"\n✅ 降噪完成!")
        print(f"📊 输出形状: {result.shape}")
        print(f"📈 统计信息:")
        print(f"   最小值: {np.min(result):.6f}")
        print(f"   最大值: {np.max(result):.6f}")
        print(f"   平均值: {np.mean(result):.6f}")
        print(f"   标准差: {np.std(result):.6f}")
    else:
        print("❌ 降噪失败!") 