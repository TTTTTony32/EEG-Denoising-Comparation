import numpy as np

def insert_vector_at_position(vector, target_shape, position):
    """
    将一个向量插入到指定形状矩阵的指定位置，其他位置为0
    
    Args:
        vector: 输入向量，形状为(512,)
        target_shape: 目标矩阵形状，如(3740, 512)
        position: 插入位置的行索引，如178
    
    Returns:
        result_matrix: 结果矩阵，形状为target_shape
    """
    # 检查输入向量的形状
    if len(vector.shape) != 1:
        raise ValueError("输入必须是一维向量")
    
    # 检查向量长度是否与目标矩阵的列数匹配
    if vector.shape[0] != target_shape[1]:
        raise ValueError(f"向量长度 {vector.shape[0]} 与目标矩阵列数 {target_shape[1]} 不匹配")
    
    # 检查位置是否有效
    if position < 0 or position >= target_shape[0]:
        raise ValueError(f"位置 {position} 超出范围 [0, {target_shape[0]-1}]")
    
    # 创建全零矩阵
    result_matrix = np.zeros(target_shape)
    
    # 在指定位置插入向量
    result_matrix[position, :] = vector
    
    return result_matrix

# 示例使用
if __name__ == "__main__":
    
    sample_vector = np.load('approaches/denoised_eegdfus_dfus.npy')  # 或者其他方式加载数据
    
    # 目标形状和位置
    target_shape = (3740, 512)
    insert_position = 178
    
    # 执行插入操作
    result = insert_vector_at_position(sample_vector, target_shape, insert_position)
    
    # 验证结果
    print(f"结果矩阵形状: {result.shape}")
    print(f"第{insert_position}行是否为原向量: {np.array_equal(result[insert_position, :], sample_vector)}")
    print(f"其他行是否全为0: {np.all(result[:insert_position, :] == 0) and np.all(result[insert_position+1:, :] == 0)}")
    
    # 可选: 保存结果
    np.save('./denoised_eegdfus_vec.npy', result)
    print("操作完成！")
