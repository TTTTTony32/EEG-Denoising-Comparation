"""
Author: wolider wong
Date: 2024-1-11
Description: 在测试集上进行降噪并输出完整的(2700,512)形状的numpy矩阵
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AbstractDenoiser
from utils.train_valid_utils import get_config, init_model, load_dataset
from accelerate import Accelerator


def denoise_test_data(config_path, weight_path, output_path="./denoised_output.npy"):
    """
    在测试集上进行降噪并输出完整的numpy矩阵
    
    Args:
        config_path: 配置文件路径
        weight_path: 模型权重文件路径
        output_path: 输出文件路径
    """
    print(f"加载配置文件: {config_path}")
    config = get_config(config_path)
    
    # 设置权重路径
    config["model"]["weight_path"] = weight_path
    
    # 初始化模型
    print("初始化模型...")
    model = init_model(config)
    
    # 加载测试数据
    print("加载测试数据...")
    test_x = np.load("./datasets/test_input.npy")
    test_y = np.load("./datasets/test_output.npy")
    
    # 检查数据形状
    print(f"测试输入数据形状: {test_x.shape}")
    print(f"测试输出数据形状: {test_y.shape}")
    
    # 准备数据
    test_std = np.ones((test_x.shape[0], 1))
    test_data = {
        'train_x': test_x,  # 这里用测试数据作为训练数据（只是为了兼容接口）
        'train_y': test_y,  # 这里用测试数据作为训练数据（只是为了兼容接口）
        'test_x': test_x,   # 实际测试输入
        'test_y': test_y,   # 实际测试输出
    }
    
    # 加载数据集
    _, test_dataset = load_dataset(config, custom_data=test_data)
    
    # 设置数据加载器
    batch_size = config["train"]["batch_size"]
    data_loader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    # 初始化Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # 准备模型和数据加载器
    model, data_loader_test = accelerator.prepare(model, data_loader_test)
    
    # 设置为评估模式
    model.eval()
    
    # 存储所有预测结果
    all_predictions = []
    
    print("开始降噪处理...")
    with torch.no_grad():
        for batch in tqdm(data_loader_test, desc="降噪处理"):
            inputs = batch["y"]  # 含噪声的输入
            labels = batch["x"]  # 干净信号（ground truth）
            std = batch["std"]   # 标准差
            
            # 前向传播
            outputs, _ = model(inputs, labels)
            
            # 反归一化（如果需要的话）
            # outputs = outputs * std
            
            # 收集所有设备上的结果
            outputs = accelerator.gather_for_metrics(outputs)
            
            # 转换为numpy并添加到结果列表
            outputs_np = outputs.cpu().detach().numpy()
            all_predictions.append(outputs_np)
    
    # 合并所有批次的结果
    print("合并结果...")
    denoised_output = np.concatenate(all_predictions, axis=0)
    
    # 确保输出形状正确
    if denoised_output.shape != test_x.shape:
        print(f"警告: 输出形状 {denoised_output.shape} 与期望形状 {test_x.shape} 不匹配")
        # 如果形状不匹配，可能需要调整
        if denoised_output.shape[0] > test_x.shape[0]:
            denoised_output = denoised_output[:test_x.shape[0], :]
        elif denoised_output.shape[0] < test_x.shape[0]:
            # 如果输出样本数少于输入，可能需要特殊处理
            print("错误: 输出样本数少于输入样本数")
            return None
    
    print(f"降噪完成! 输出形状: {denoised_output.shape}")
    
    # 保存结果
    print(f"保存结果到: {output_path}")
    np.save(output_path, denoised_output)
    
    # 同时保存为txt格式（可选）
    txt_path = output_path.replace('.npy', '.txt')
    np.savetxt(txt_path, denoised_output, fmt='%.6f')
    print(f"同时保存为txt格式: {txt_path}")
    
    return denoised_output


def main():
    """主函数"""
    # 配置文件路径
    config_path = "config/retnet/config.yml"
    
    # 模型权重文件路径 - 请根据实际情况修改
    # 可以是训练好的最佳模型权重
    weight_path = "./results/2024_01_11_15_DiR_4_EOG_pathch16_mini_seq32_hidden_dim512_layer_1_EMG/weight/best.pth"
    
    # 检查权重文件是否存在
    if not os.path.exists(weight_path):
        print(f"警告: 权重文件不存在: {weight_path}")
        print("请检查权重文件路径，或者使用其他训练好的模型权重文件")
        
        # 尝试查找其他权重文件
        results_dir = "./results"
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    if file.endswith('.pth'):
                        potential_weight = os.path.join(root, file)
                        print(f"找到权重文件: {potential_weight}")
                        weight_path = potential_weight
                        break
                if weight_path != "":
                    break
    
    # 输出文件路径
    output_path = "./denoised_test_output.npy"
    
    # 执行降噪
    result = denoise_test_data(config_path, weight_path, output_path)
    
    if result is not None:
        print(f"\n降噪完成!")
        print(f"输入数据形状: (2700, 512)")
        print(f"输出数据形状: {result.shape}")
        print(f"结果已保存到: {output_path}")
        
        # 显示一些统计信息
        print(f"\n统计信息:")
        print(f"最小值: {np.min(result):.6f}")
        print(f"最大值: {np.max(result):.6f}")
        print(f"平均值: {np.mean(result):.6f}")
        print(f"标准差: {np.std(result):.6f}")
    else:
        print("降噪失败!")


if __name__ == "__main__":
    main() 