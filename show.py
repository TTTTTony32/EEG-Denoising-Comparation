import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"

# 读取数据
data_clean = np.load('prepared_data/test_output.npy')
data_noised = np.load('prepared_data/test_input.npy')


# 随机选择1个样本
random_index = np.random.choice(data_clean.shape[0], 1, replace=False)[0]
sample1 = data_clean[random_index]
sample2 = data_noised[random_index]

# 归一化时间轴
sampling_rate = 256  # 采样频率
time = np.linspace(0, len(sample1) / sampling_rate, len(sample1))

# 绘制样本
plt.figure(figsize=(10, 5))

# 绘制两个样本在一个框里
plt.plot(time, sample2, label='Noised', color=(235/255, 160/255, 55/255))
plt.plot(time, sample1, label='Clean', color=(0/255, 166/255, 156/255))

plt.title(f'Segment {random_index}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, which='both', linestyle='--', color=(187/255, 187/255, 187/255))

plt.tight_layout()

# 保存图片
plt.savefig('sample_compare_plot.pdf', dpi=300)

