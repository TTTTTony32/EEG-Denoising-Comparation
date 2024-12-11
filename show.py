import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"

# 读取数据
data = np.load('datasets/EEG_all_epochs.npy')

# 显示数据结构
print("Data shape:", data.shape)
print("Data type:", data.dtype)

# 随机选择1个样本
random_index = np.random.choice(data.shape[0], 1, replace=False)[0]
# sample = data[random_index]
sample = data[2579]

# 归一化时间轴
sampling_rate = 256  # 采样频率
time = np.linspace(0, len(sample) / sampling_rate, len(sample))

# 绘制样本
plt.plot(time, sample)
# plt.title(f'Sample {random_index}')
plt.title(f'Segment 2579')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 保存图片
plt.savefig('sample_plot.pdf', dpi=300)

