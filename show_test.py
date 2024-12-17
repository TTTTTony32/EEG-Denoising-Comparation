import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"

# 读取数据
data = np.load('prepared_data/test_output.npy')
data_1 = data[100:104,:]
data_d = data_1.ravel()
print(data_d.shape)
# 归一化时间轴
sampling_rate = 256  # 采样频率
time = np.linspace(0, len(data_d) / sampling_rate, len(data_d))

# 绘制样本
plt.figure(figsize=(10, 5))

# 绘制两个样本在一个框里
plt.plot(time, data_d, label='Clean', color=(0/255, 166/255, 156/255))

plt.tight_layout()

# 保存图片
plt.show()

