'''
简化的训练脚本 - 直接使用预处理好的数据
不使用10折交叉验证，直接训练
'''
import argparse, torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from opts import get_opts
from audtorch.metrics.functional import pearsonr
import math

import os
from models import *
from loss import denoise_loss_mse
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from tools import pick_models

loss_type = "feature+cls" 
if loss_type == "feature":
    w_c = 0
    w_f = 0.05
elif loss_type == "cls":
    w_c = 0.05
    w_f = 0
elif loss_type == "feature+cls":
    w_f = 0.05
    w_c = 0.05

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=-1)
    PN = np.sum(np.square((predict - truth)), axis=-1)
    ratio = PS / PN
    return torch.from_numpy(10 * np.log10(ratio))

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def load_custom_data(data_path):
    """
    加载自定义格式的数据
    """
    train_input = np.load(os.path.join(data_path, 'train_input.npy'))
    train_output = np.load(os.path.join(data_path, 'train_output.npy'))
    test_input = np.load(os.path.join(data_path, 'test_input.npy'))
    test_output = np.load(os.path.join(data_path, 'test_output.npy'))
    val_input = np.load(os.path.join(data_path, 'val_input.npy'))
    val_output = np.load(os.path.join(data_path, 'val_output.npy'))
    
    return train_input, train_output, val_input, val_output, test_input, test_output

class CustomDataset:
    def __init__(self, input_data, output_data, batch_size=128):
        self.input_data = input_data
        self.output_data = output_data
        self.batch_size = batch_size
        self.num_samples = input_data.shape[0]

    def len(self):
        return math.ceil(self.num_samples / self.batch_size)

    def get_batch(self, batch_id):
        start_id = batch_id * self.batch_size
        end_id = min((batch_id + 1) * self.batch_size, self.num_samples)
        
        x_batch = self.input_data[start_id:end_id]
        y_batch = self.output_data[start_id:end_id]
        
        return x_batch, y_batch

    def shuffle(self):
        indices = np.random.permutation(self.num_samples)
        self.input_data = self.input_data[indices]
        self.output_data = self.output_data[indices]

def train(opts, model, train_log_dir, val_log_dir, data_save_path):
    # 加载数据
    train_input, train_output, val_input, val_output, test_input, test_output = load_custom_data(opts.data_path)
    
    print(f"训练数据: {train_input.shape}")
    print(f"验证数据: {val_input.shape}")
    print(f"测试数据: {test_input.shape}")
    
    # 创建数据集
    train_data = CustomDataset(train_input, train_output, opts.batch_size)
    val_data = CustomDataset(val_input, val_output, opts.batch_size)
    test_data = CustomDataset(test_input, test_output, opts.batch_size)
    
    model_d = Discriminator().to('cuda:0')
    
    model_d.apply(weights_init)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-8)
    optimizer_D = torch.optim.Adam(model_d.parameters(), lr=0.0001)
        
    best_mse = 100
    if opts.save_result:
        train_summary_writer = SummaryWriter(train_log_dir)
        val_summary_writer = SummaryWriter(val_log_dir)
        f = open(data_save_path + "result.txt", "a+")
    
    for epoch in range(opts.epochs):
        model.train()
        model_d.train()
        losses = []
        for batch_id in trange(train_data.len()):
            x_t, y_t = train_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)
            
            y_original = y_t
            if batch_id % 1 == 0:
                p_t = model(x_t).view(x_t.shape[0], -1)
                fake_y, _, _, _ = model_d(p_t.unsqueeze(dim=1))
                real_y, _, _, _ = model_d(y_t.unsqueeze(dim=1))
                
                d_loss = 0.5 * (torch.mean((fake_y) ** 2)) + 0.5 * (torch.mean((real_y - 1) ** 2))
                
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()
            
            if batch_id % 1 == 0:
                p_t = model(x_t).view(x_t.shape[0], -1)

                fake_y, _, fake_feature2, _ = model_d(p_t.unsqueeze(dim=1))
                _, _, true_feature2, _ = model_d(y_t.unsqueeze(dim=1))

                y_t = y_original
                loss = denoise_loss_mse(p_t, y_t)
                
                if loss_type == "cls":
                    g_loss = loss + w_c * (torch.mean((fake_y - 1) ** 2)) 
                elif loss_type == "feature": 
                    g_loss = loss + w_f * denoise_loss_mse(fake_feature2, true_feature2)
                elif loss_type == "feature+cls": 
                    g_loss = loss + w_f * denoise_loss_mse(fake_feature2, true_feature2) + w_c * (torch.mean((fake_y - 1) ** 2))  

                optimizer_D.zero_grad()
                optimizer.zero_grad()
                g_loss.backward()
                optimizer.step()
                    
                losses.append(g_loss.detach())
                
        train_data.shuffle()
        train_loss = torch.stack(losses).mean().item()

        if opts.save_result:
            train_summary_writer.add_scalar("Train loss", train_loss, epoch)
        
        # 验证
        model.eval()
        losses = []
        for batch_id in range(val_data.len()):
            x_t, y_t = val_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)
            
            with torch.no_grad():
                p_t = model(x_t).view(x_t.shape[0], -1)
                loss = ((p_t - y_t) ** 2).mean(dim=-1).sqrt().detach()
                losses.append(loss)
        val_mse = torch.cat(losses, dim=0).mean().item()
        val_summary_writer.add_scalar("Val loss", val_mse, epoch)
        
        # 测试
        model.eval()
        losses = []
        single_acc, single_snr = [], []
        clean_data, output_data, input_data = [], [], []
        correct_d, sum_d = 0, 0
        for batch_id in range(test_data.len()):
            x_t, y_t = test_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)

            with torch.no_grad():
                p_t = model(x_t).view(x_t.shape[0], -1)
                loss = (((p_t - y_t) ** 2).mean(dim=-1).sqrt() / (y_t ** 2).mean(dim=-1).sqrt()).detach()
                losses.append(loss.detach())
                single_acc.append(pearsonr(p_t, y_t))
                single_snr.append(cal_SNR(p_t, y_t))
                
                p_t = model(x_t).view(x_t.shape[0], -1)
                
                fake_y, _, _, _ = model_d(p_t.unsqueeze(dim=1))
                real_y, _, _, _ = model_d(y_t.unsqueeze(dim=1))
                
                correct_d += torch.sum(torch.where(fake_y < 0.5, 1, 0)) + torch.sum(torch.where(real_y > 0.5, 1, 0))
                sum_d += p_t.shape[0] * 2
                    
            output_data.append(p_t.cpu().numpy()), clean_data.append(y_t.cpu().numpy()), input_data.append(x_t.cpu().numpy())
                    
        test_rrmse = torch.cat(losses, dim=0).mean().item()
        sum_acc = torch.cat(single_acc, dim=0).mean().item()
        sum_snr = torch.cat(single_snr, dim=0).mean().item()
        
        val_summary_writer.add_scalar("test rrmse", test_rrmse, epoch)
        
        # 保存最佳结果
        if val_mse < best_mse:
            best_mse = val_mse
            best_acc = sum_acc
            best_snr = sum_snr
            best_rrmse = test_rrmse
            print("Save best result")
            f.write("Save best result \n")
            val_summary_writer.add_scalar("best rrmse", best_mse, epoch)
            if opts.save_result:
                # 将列表中的数组连接成一个大的数组
                input_data_concat = np.concatenate(input_data, axis=0)
                output_data_concat = np.concatenate(output_data, axis=0)
                clean_data_concat = np.concatenate(clean_data, axis=0)
                
                np.save(f"{data_save_path}/best_input_data.npy", input_data_concat)
                np.save(f"{data_save_path}/best_output_data.npy", output_data_concat)
                np.save(f"{data_save_path}/best_clean_data.npy", clean_data_concat)
                torch.save(model, f"{data_save_path}/best_{opts.denoise_network}.pth")

        print('correct_d: {:3d}, sum_d:{:.4f}, acc_d:{}'.format(correct_d.cpu().numpy(), sum_d, correct_d.cpu().numpy()/sum_d*1.0))
        print('epoch: {:3d}, train_loss:{:.4f}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(epoch, train_loss, test_rrmse, sum_acc, sum_snr))
        f.write('epoch: {:3d}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(epoch, test_rrmse, sum_acc, sum_snr) + "\n")

    print(f"\n训练完成！")
    print(f"最佳结果 - RRMSE: {best_rrmse:.4f}, ACC: {best_acc:.4f}, SNR: {best_snr:.4f}")
    
    if opts.save_result:
        np.save(f"{data_save_path}/last_input_data.npy", test_data.input_data)
        # 将列表中的数组连接成一个大的数组
        output_data_concat = np.concatenate(output_data, axis=0)
        clean_data_concat = np.concatenate(clean_data, axis=0)
        
        np.save(f"{data_save_path}/last_output_data.npy", output_data_concat)
        np.save(f"{data_save_path}/last_clean_data.npy", clean_data_concat)
        torch.save(model, f"{data_save_path}/last_{opts.denoise_network}.pth")

if __name__ == '__main__':
    opts = get_opts()
    np.random.seed(0)
    opts.epochs = 200        # 训练轮数
    opts.depth = 6
    opts.noise_type = 'Custom'     # 自定义噪声类型
    opts.denoise_network = 'GCTNet'
    opts.data_path = "./data/"     # 你的数据路径

    opts.save_path = "./results/{}/{}/".format(opts.noise_type, opts.denoise_network)

    print("开始训练...")
    model = pick_models(opts, data_num=512)
    print(f"使用模型: {opts.denoise_network}")
    
    foldername = '{}_{}_{}_{}_{}'.format(opts.denoise_network, opts.noise_type, opts.epochs, w_c, w_f)
        
    train_log_dir = opts.save_path +'/'+foldername +'/'+ '/train'
    val_log_dir = opts.save_path +'/'+foldername +'/'+ '/test'
    data_save_path = opts.save_path +'/'+foldername +'/'
    
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    
    if not os.path.exists(val_log_dir):
        os.makedirs(val_log_dir)
    
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    train(opts, model, train_log_dir, val_log_dir, data_save_path) 