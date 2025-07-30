import torch
import numpy as np
import yaml
import argparse
import os
from DDPM import DDPM
from denoising_model_eegdnet import DualBranchDenoisingModel

def load_model(config_path, model_path, device):
    """加载训练好的模型"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    base_model = DualBranchDenoisingModel(config['train']['feats']).to(device)
    model = DDPM(base_model, config, device)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def denoise_signals(model, noisy_signals, device, batch_size=32):
    """对信号进行降噪处理"""
    model.eval()
    denoised_signals = []
    
    # 确保输入格式正确
    if len(noisy_signals.shape) == 2:
        # 如果输入是 (N, 512)，转换为 (N, 1, 512)
        noisy_signals = noisy_signals.unsqueeze(1)
    
    num_samples = noisy_signals.shape[0]
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # 获取当前批次
            end_idx = min(i + batch_size, num_samples)
            batch_signals = noisy_signals[i:end_idx].to(device)
            
            # 使用DDPM进行降噪
            denoised_batch = model.denoising(batch_signals)
            
            # 移除通道维度，得到 (batch_size, 512)
            denoised_batch = denoised_batch.squeeze(1)
            
            denoised_signals.append(denoised_batch.cpu())
    
    # 合并所有批次的结果
    denoised_signals = torch.cat(denoised_signals, dim=0)
    
    return denoised_signals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml", help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--input_path", type=str, required=True, help="输入测试数据路径 (.npy)")
    parser.add_argument("--output_path", type=str, required=True, help="输出降噪结果路径 (.npy)")
    parser.add_argument("--device", default="cuda:0", help="设备")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"输入文件不存在: {args.input_path}")
    
    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    model = load_model(args.config, args.model_path, args.device)
    print("模型加载完成")
    
    # 加载测试数据
    print(f"正在加载测试数据: {args.input_path}")
    test_data = np.load(args.input_path)
    print(f"测试数据形状: {test_data.shape}")
    
    # 检查数据形状
    if len(test_data.shape) != 2:
        raise ValueError(f"输入数据应该是2维数组 (N, 512)，当前形状: {test_data.shape}")
    
    if test_data.shape[1] != 512:
        raise ValueError(f"输入数据的第二个维度应该是512，当前: {test_data.shape[1]}")
    
    # 转换为tensor
    test_tensor = torch.FloatTensor(test_data)
    
    # 进行降噪处理
    print("开始降噪处理...")
    denoised_signals = denoise_signals(model, test_tensor, args.device, args.batch_size)
    
    # 转换为numpy数组
    denoised_numpy = denoised_signals.numpy()
    
    # 保存结果
    print(f"保存降噪结果到: {args.output_path}")
    np.save(args.output_path, denoised_numpy)
    
    print(f"处理完成！")
    print(f"输入形状: {test_data.shape}")
    print(f"输出形状: {denoised_numpy.shape}")
    print(f"输出范围: [{denoised_numpy.min():.4f}, {denoised_numpy.max():.4f}]")

if __name__ == "__main__":
    main() 