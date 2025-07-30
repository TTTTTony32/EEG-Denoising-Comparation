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

def denoise_single_sample(model, noisy_signal, device):
    """对单个样本进行降噪处理"""
    model.eval()
    
    # 确保输入格式正确
    if len(noisy_signal.shape) == 1:
        # 如果输入是 (512,)，转换为 (1, 1, 512)
        noisy_signal = noisy_signal.unsqueeze(0).unsqueeze(0)
    elif len(noisy_signal.shape) == 2:
        # 如果输入是 (1, 512)，转换为 (1, 1, 512)
        noisy_signal = noisy_signal.unsqueeze(0)
    
    with torch.no_grad():
        # 使用DDPM进行降噪
        denoised_signal = model.denoising(noisy_signal.to(device))
        
        # 移除批次和通道维度，得到 (512,)
        denoised_signal = denoised_signal.squeeze(0).squeeze(0)
    
    return denoised_signal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml", help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--input_path", type=str, required=True, help="输入数据路径 (.npy)")
    parser.add_argument("--output_path", type=str, required=True, help="输出结果路径 (.npy)")
    parser.add_argument("--sample_idx", type=int, default=0, help="要处理的样本索引")
    parser.add_argument("--device", default="cuda:0", help="设备")
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
    
    # 加载数据
    print(f"正在加载数据: {args.input_path}")
    data = np.load(args.input_path)
    print(f"数据形状: {data.shape}")
    
    # 检查数据形状
    if len(data.shape) != 2:
        raise ValueError(f"输入数据应该是2维数组 (N, 512)，当前形状: {data.shape}")
    
    if data.shape[1] != 512:
        raise ValueError(f"输入数据的第二个维度应该是512，当前: {data.shape[1]}")
    
    # 检查样本索引
    if args.sample_idx >= data.shape[0]:
        raise ValueError(f"样本索引 {args.sample_idx} 超出范围，数据只有 {data.shape[0]} 个样本")
    
    # 提取单个样本
    single_sample = data[args.sample_idx]
    print(f"处理样本 {args.sample_idx}: 形状 {single_sample.shape}")
    
    # 转换为tensor
    sample_tensor = torch.FloatTensor(single_sample)
    
    # 进行降噪处理
    print("开始降噪处理...")
    denoised_sample = denoise_single_sample(model, sample_tensor, args.device)
    
    # 转换为numpy数组
    denoised_numpy = denoised_sample.cpu().numpy()
    
    # 保存结果
    print(f"保存降噪结果到: {args.output_path}")
    np.save(args.output_path, denoised_numpy)
    
    print(f"处理完成！")
    print(f"输入样本形状: {single_sample.shape}")
    print(f"输出样本形状: {denoised_numpy.shape}")
    print(f"输入范围: [{single_sample.min():.4f}, {single_sample.max():.4f}]")
    print(f"输出范围: [{denoised_numpy.min():.4f}, {denoised_numpy.max():.4f}]")

if __name__ == "__main__":
    main() 