'''
数据格式验证脚本
检查你的数据文件是否符合要求
'''
import numpy as np
import os

def check_data_format(data_path):
    """
    检查数据格式是否正确
    """
    required_files = [
        'train_input.npy',
        'train_output.npy', 
        'test_input.npy',
        'test_output.npy',
        'val_input.npy',
        'val_output.npy'
    ]
    
    print("检查数据文件...")
    
    # 检查文件是否存在
    for file_name in required_files:
        file_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_path):
            print(f"❌ 缺少文件: {file_name}")
            return False
        else:
            print(f"✅ 找到文件: {file_name}")
    
    # 检查数据形状
    print("\n检查数据形状...")
    try:
        train_input = np.load(os.path.join(data_path, 'train_input.npy'))
        train_output = np.load(os.path.join(data_path, 'train_output.npy'))
        test_input = np.load(os.path.join(data_path, 'test_input.npy'))
        test_output = np.load(os.path.join(data_path, 'test_output.npy'))
        val_input = np.load(os.path.join(data_path, 'val_input.npy'))
        val_output = np.load(os.path.join(data_path, 'val_output.npy'))
        
        print(f"train_input.shape: {train_input.shape}")
        print(f"train_output.shape: {train_output.shape}")
        print(f"test_input.shape: {test_input.shape}")
        print(f"test_output.shape: {test_output.shape}")
        print(f"val_input.shape: {val_input.shape}")
        print(f"val_output.shape: {val_output.shape}")
        
        # 检查形状是否一致
        if train_input.shape[1] != 512 or train_output.shape[1] != 512:
            print("❌ 数据长度不是512个采样点")
            return False
            
        if train_input.shape[0] != train_output.shape[0]:
            print("❌ train_input和train_output的样本数不匹配")
            return False
            
        if test_input.shape[0] != test_output.shape[0]:
            print("❌ test_input和test_output的样本数不匹配")
            return False
            
        if val_input.shape[0] != val_output.shape[0]:
            print("❌ val_input和val_output的样本数不匹配")
            return False
            
        print("✅ 数据形状检查通过")
        
        # 检查数据类型
        print("\n检查数据类型...")
        print(f"train_input.dtype: {train_input.dtype}")
        print(f"train_output.dtype: {train_output.dtype}")
        
        if train_input.dtype != np.float32 and train_input.dtype != np.float64:
            print("⚠️  建议将输入数据转换为float32或float64类型")
            
        if train_output.dtype != np.float32 and train_output.dtype != np.float64:
            print("⚠️  建议将输出数据转换为float32或float64类型")
            
        # 检查数据范围
        print("\n检查数据范围...")
        print(f"train_input范围: [{train_input.min():.4f}, {train_input.max():.4f}]")
        print(f"train_output范围: [{train_output.min():.4f}, {train_output.max():.4f}]")
        
        # 检查是否有NaN或无穷大值
        if np.isnan(train_input).any() or np.isinf(train_input).any():
            print("❌ train_input包含NaN或无穷大值")
            return False
            
        if np.isnan(train_output).any() or np.isinf(train_output).any():
            print("❌ train_output包含NaN或无穷大值")
            return False
            
        print("✅ 数据质量检查通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        return False

if __name__ == '__main__':
    data_path = "./data/"  # 你的数据路径
    
    if check_data_format(data_path):
        print("\n🎉 数据格式检查通过！可以开始训练了。")
        print("\n使用方法:")
        print("python train_custom_data.py")
    else:
        print("\n❌ 数据格式有问题，请检查并修正。") 