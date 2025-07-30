# 自定义数据训练指南

## 数据格式要求

你的数据需要包含以下6个 `.npy` 文件，放在 `./data/` 目录下：

- `train_input.npy` - 训练集输入数据 (3750, 512)
- `train_output.npy` - 训练集输出数据 (3750, 512)  
- `test_input.npy` - 测试集输入数据 (3750, 512)
- `test_output.npy` - 测试集输出数据 (3750, 512)
- `val_input.npy` - 验证集输入数据 (3750, 512)
- `val_output.npy` - 验证集输出数据 (3750, 512)

其中：
- `3750` 是片段数量
- `512` 是每个片段的采样点数
- `input` 是带噪声的信号
- `output` 是干净的信号（目标输出）

## 使用步骤

### 1. 检查数据格式

首先运行数据验证脚本，确保你的数据格式正确：

```bash
python check_data.py
```

如果看到 "🎉 数据格式检查通过！" 就说明数据格式正确。

### 2. 选择训练脚本

有两个训练脚本可供选择：

#### 选项1：简化训练脚本（推荐）
```bash
python train_simple.py
```
- 不使用10折交叉验证
- 直接使用你的训练/验证/测试数据
- 训练更快，适合快速验证

#### 选项2：完整训练脚本
```bash
python train_custom_data.py
```
- 使用10折交叉验证
- 训练时间更长，但结果更稳定

### 3. 修改训练参数

在训练脚本中，你可以修改以下参数：

```python
opts.epochs = 200        # 训练轮数
opts.batch_size = 128    # 批次大小
opts.depth = 6          # 网络深度
opts.denoise_network = 'GCTNet'  # 网络类型
opts.data_path = "./data/"       # 数据路径
```

### 4. 运行训练

```bash
python train_simple.py
```

训练过程中会显示：
- 训练损失
- 验证损失  
- 测试指标（RRMSE, ACC, SNR）
- 判别器准确率

### 5. 查看结果

训练完成后，结果保存在：
```
./results/Custom/GCTNet/GCTNet_Custom_200_0.05_0.05/
├── best_GCTNet.pth          # 最佳模型
├── best_input_data.npy      # 最佳预测的输入数据
├── best_output_data.npy     # 最佳预测的输出数据
├── best_clean_data.npy      # 最佳预测的干净数据
├── result.txt               # 训练日志
└── train/                   # TensorBoard日志
```

## 数据预处理建议

如果你的数据还没有预处理，建议：

1. **标准化**：对每个信号片段进行标准化
2. **数据类型**：确保数据类型为 `float32` 或 `float64`
3. **数据质量**：检查是否有NaN或无穷大值
4. **数据平衡**：确保训练/验证/测试集的分布相似

## 常见问题

### Q: 数据形状不匹配怎么办？
A: 确保所有数据文件的形状都是 `(样本数, 512)`，且对应的input和output文件样本数相同。

### Q: 训练很慢怎么办？
A: 可以：
- 减少 `epochs` 数量
- 增加 `batch_size`（如果GPU内存足够）
- 使用 `train_simple.py` 而不是 `train_custom_data.py`

### Q: 内存不足怎么办？
A: 可以：
- 减少 `batch_size`
- 减少 `depth` 参数
- 使用更小的网络

### Q: 如何修改损失函数权重？
A: 在脚本开头修改：
```python
loss_type = "feature+cls"  # 可选: "feature", "cls", "feature+cls"
w_f = 0.05  # 特征损失权重
w_c = 0.05  # 分类损失权重
```

## 模型说明

- **GCTNet**: 主要的去噪网络
- **Discriminator**: 对抗训练的判别器
- **损失函数**: MSE + 特征损失 + 分类损失

训练使用对抗学习框架，包含生成器和判别器的交替训练。 