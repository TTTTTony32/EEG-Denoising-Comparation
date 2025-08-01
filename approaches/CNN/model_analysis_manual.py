"""
手动分析EEG降噪模型的参数量和FLOPs
基于Network_structure.py中的模型定义
"""

def analyze_simple_CNN(datanum=512):
    """
    分析simple_CNN模型
    输入: (batch_size, datanum, 1)
    """
    print("="*60)
    print("SIMPLE_CNN 模型分析")
    print("="*60)
    print(f"输入维度: (batch_size, {datanum}, 1)")
    
    params = 0
    flops = 0
    
    # Layer 1: Conv1D(1->64, kernel=3, stride=1, padding=same)
    in_ch, out_ch, kernel = 1, 64, 3
    conv1_params = in_ch * out_ch * kernel + out_ch  # weights + bias
    conv1_flops = in_ch * out_ch * kernel * datanum
    params += conv1_params
    flops += conv1_flops
    print(f"Conv1D-1: 参数={conv1_params:,}, FLOPs={conv1_flops:,}")
    
    # BatchNorm + ReLU (BatchNorm参数: 2*channels)
    bn1_params = 2 * out_ch
    params += bn1_params
    print(f"BatchNorm-1: 参数={bn1_params}")
    
    # Layer 2: Conv1D(64->64, kernel=3)
    in_ch, out_ch, kernel = 64, 64, 3
    conv2_params = in_ch * out_ch * kernel + out_ch
    conv2_flops = in_ch * out_ch * kernel * datanum
    params += conv2_params
    flops += conv2_flops
    print(f"Conv1D-2: 参数={conv2_params:,}, FLOPs={conv2_flops:,}")
    
    bn2_params = 2 * out_ch
    params += bn2_params
    print(f"BatchNorm-2: 参数={bn2_params}")
    
    # Layer 3: Conv1D(64->64, kernel=3)
    conv3_params = conv2_params
    conv3_flops = conv2_flops
    params += conv3_params
    flops += conv3_flops
    print(f"Conv1D-3: 参数={conv3_params:,}, FLOPs={conv3_flops:,}")
    
    bn3_params = bn2_params
    params += bn3_params
    print(f"BatchNorm-3: 参数={bn3_params}")
    
    # Layer 4: Conv1D(64->64, kernel=3)
    conv4_params = conv2_params
    conv4_flops = conv2_flops
    params += conv4_params
    flops += conv4_flops
    print(f"Conv1D-4: 参数={conv4_params:,}, FLOPs={conv4_flops:,}")
    
    bn4_params = bn2_params
    params += bn4_params
    print(f"BatchNorm-4: 参数={bn4_params}")
    
    # Flatten + Dense layer: (64*datanum) -> datanum
    dense_params = (64 * datanum) * datanum + datanum
    dense_flops = (64 * datanum) * datanum
    params += dense_params
    flops += dense_flops
    print(f"Dense: 参数={dense_params:,}, FLOPs={dense_flops:,}")
    
    print(f"\n总参数量: {params:,}")
    print(f"总FLOPs: {flops:,}")
    print(f"总FLOPs (GFLOPs): {flops/1e9:.3f}")
    
    return params, flops

def analyze_RNN_lstm(datanum=512):
    """
    分析RNN_lstm模型
    输入: (batch_size, datanum, 1)
    """
    print("\n" + "="*60)
    print("RNN_LSTM 模型分析")
    print("="*60)
    print(f"输入维度: (batch_size, {datanum}, 1)")
    
    params = 0
    flops = 0
    
    # LSTM layer: input_size=1, hidden_size=1, return_sequences=True
    input_size = 1
    hidden_size = 1
    
    # LSTM参数计算: 4 * (input_size + hidden_size + 1) * hidden_size
    # 4个门: forget, input, candidate, output
    lstm_params = 4 * (input_size + hidden_size + 1) * hidden_size
    # LSTM FLOPs: 约为 4 * hidden_size * (input_size + hidden_size) * sequence_length
    lstm_flops = 4 * hidden_size * (input_size + hidden_size) * datanum
    
    params += lstm_params
    flops += lstm_flops
    print(f"LSTM: 参数={lstm_params}, FLOPs={lstm_flops:,}")
    
    # Flatten: (datanum, 1) -> (datanum,)
    # No parameters, no significant FLOPs
    
    # Dense layer 1: datanum -> datanum
    dense1_params = datanum * datanum + datanum
    dense1_flops = datanum * datanum
    params += dense1_params
    flops += dense1_flops
    print(f"Dense-1: 参数={dense1_params:,}, FLOPs={dense1_flops:,}")
    
    # Dense layer 2: datanum -> datanum
    dense2_params = dense1_params
    dense2_flops = dense1_flops
    params += dense2_params
    flops += dense2_flops
    print(f"Dense-2: 参数={dense2_params:,}, FLOPs={dense2_flops:,}")
    
    # Dense layer 3: datanum -> datanum
    dense3_params = dense1_params
    dense3_flops = dense1_flops
    params += dense3_params
    flops += dense3_flops
    print(f"Dense-3: 参数={dense3_params:,}, FLOPs={dense3_flops:,}")
    
    print(f"\n总参数量: {params:,}")
    print(f"总FLOPs: {flops:,}")
    print(f"总FLOPs (MFLOPs): {flops/1e6:.3f}")
    
    return params, flops

def analyze_TwobythreeR_CNN(datanum=512):
    """
    分析TwobythreeR_CNN模型 (PyTorch)
    输入: (batch_size, datanum) -> reshape to (batch_size, 1, datanum)
    """
    print("\n" + "="*60)
    print("TWOBYTHEER_CNN 模型分析")
    print("="*60)
    print(f"输入维度: (batch_size, 1, {datanum})")
    
    params = 0
    flops = 0
    
    # net0: 初始特征提取
    print("\n--- net0 (初始特征提取) ---")
    # Conv1d(1->32, kernel=5)
    conv_params = 1 * 32 * 5 + 32
    conv_flops = 1 * 32 * 5 * datanum
    params += conv_params
    flops += conv_flops
    print(f"Conv1d(1->32, k=5): 参数={conv_params}, FLOPs={conv_flops:,}")
    
    # Conv1d(32->32, kernel=5)
    conv_params = 32 * 32 * 5 + 32
    conv_flops = 32 * 32 * 5 * datanum
    params += conv_params
    flops += conv_flops
    print(f"Conv1d(32->32, k=5): 参数={conv_params:,}, FLOPs={conv_flops:,}")
    
    # net1: 第一个分支 (kernel=3)
    print("\n--- net1 (kernel=3分支) ---")
    net1_conv_ops = [
        (32, 32, 3), (32, 32, 3),  # 第一对
        (32, 16, 3), (16, 16, 3),  # 降维
        (16, 32, 3), (32, 32, 3),  # 升维
        (32, 32, 3), (32, 32, 3),  # 第二对
        (32, 16, 3), (16, 16, 3),  # 降维
        (16, 32, 3), (32, 32, 3),  # 升维
    ]
    
    net1_params = 0
    net1_flops = 0
    for in_ch, out_ch, k in net1_conv_ops:
        layer_params = in_ch * out_ch * k + out_ch
        layer_flops = in_ch * out_ch * k * datanum
        net1_params += layer_params
        net1_flops += layer_flops
    
    # net1被使用两次 (forward中out1 = self.net1(out + x))
    params += net1_params * 2
    flops += net1_flops * 2
    print(f"net1总计(x2): 参数={net1_params*2:,}, FLOPs={net1_flops*2:,}")
    
    # net2: 第二个分支 (kernel=5)
    print("\n--- net2 (kernel=5分支) ---")
    net2_conv_ops = [
        (32, 32, 5), (32, 32, 5),  # 第一对
        (32, 16, 5), (16, 16, 5),  # 降维
        (16, 32, 5), (32, 32, 5),  # 升维
        (32, 32, 5), (32, 32, 5),  # 第二对
        (32, 16, 5), (16, 16, 5),  # 降维
        (16, 32, 5), (32, 32, 5),  # 升维
    ]
    
    net2_params = 0
    net2_flops = 0
    for in_ch, out_ch, k in net2_conv_ops:
        layer_params = in_ch * out_ch * k + out_ch
        layer_flops = in_ch * out_ch * k * datanum
        net2_params += layer_params
        net2_flops += layer_flops
    
    params += net2_params * 2
    flops += net2_flops * 2
    print(f"net2总计(x2): 参数={net2_params*2:,}, FLOPs={net2_flops*2:,}")
    
    # net3: 第三个分支 (kernel=7)
    print("\n--- net3 (kernel=7分支) ---")
    net3_conv_ops = [
        (32, 32, 7), (32, 32, 7),  # 第一对
        (32, 16, 7), (16, 16, 7),  # 降维
        (16, 32, 7), (32, 32, 7),  # 升维
        (32, 32, 7), (32, 32, 7),  # 第二对
        (32, 16, 7), (16, 16, 7),  # 降维
        (16, 32, 7), (32, 32, 7),  # 升维
    ]
    
    net3_params = 0
    net3_flops = 0
    for in_ch, out_ch, k in net3_conv_ops:
        layer_params = in_ch * out_ch * k + out_ch
        layer_flops = in_ch * out_ch * k * datanum
        net3_params += layer_params
        net3_flops += layer_flops
    
    params += net3_params * 2
    flops += net3_flops * 2
    print(f"net3总计(x2): 参数={net3_params*2:,}, FLOPs={net3_flops*2:,}")
    
    # net4: 特征融合 (1x1 conv)
    print("\n--- net4 (特征融合) ---")
    # Conv1d(96->32, kernel=1) where 96 = 32*3 (concatenation)
    net4_params = 96 * 32 * 1 + 32
    net4_flops = 96 * 32 * 1 * datanum
    params += net4_params
    flops += net4_flops
    print(f"Conv1d(96->32, k=1): 参数={net4_params:,}, FLOPs={net4_flops:,}")
    
    # Linear layer: (32*datanum) -> datanum
    print("\n--- Linear layer ---")
    linear_params = (32 * datanum) * datanum + datanum
    linear_flops = (32 * datanum) * datanum
    params += linear_params
    flops += linear_flops
    print(f"Linear: 参数={linear_params:,}, FLOPs={linear_flops:,}")
    
    print(f"\n总参数量: {params:,}")
    print(f"总FLOPs: {flops:,}")
    print(f"总FLOPs (GFLOPs): {flops/1e9:.3f}")
    
    return params, flops

def main():
    """主函数：分析所有三个模型"""
    datanum = 512  # 典型的EEG数据长度
    
    print("EEG降噪模型参数量和FLOPs分析")
    print(f"假设输入数据长度: {datanum}")
    
    # 分析三个模型
    simple_cnn_params, simple_cnn_flops = analyze_simple_CNN(datanum)
    rnn_lstm_params, rnn_lstm_flops = analyze_RNN_lstm(datanum)
    twobythree_params, twobythree_flops = analyze_TwobythreeR_CNN(datanum)
    
    # 汇总比较
    print("\n" + "="*80)
    print("模型对比汇总")
    print("="*80)
    
    models = [
        ("simple_CNN", simple_cnn_params, simple_cnn_flops),
        ("RNN_lstm", rnn_lstm_params, rnn_lstm_flops),
        ("TwobythreeR_CNN", twobythree_params, twobythree_flops)
    ]
    
    print(f"{'模型名称':<20} {'参数量':<15} {'FLOPs':<15} {'GFLOPs':<10}")
    print("-" * 65)
    
    for name, params, flops in models:
        print(f"{name:<20} {params:<15,} {flops:<15,} {flops/1e9:<10.3f}")
    
    # 模型复杂度分析
    print("\n" + "="*80)
    print("模型复杂度分析")
    print("="*80)
    
    print("1. 参数量排序 (从小到大):")
    models_by_params = sorted(models, key=lambda x: x[1])
    for i, (name, params, flops) in enumerate(models_by_params, 1):
        print(f"   {i}. {name}: {params:,} 参数")
    
    print("\n2. FLOPs排序 (从小到大):")
    models_by_flops = sorted(models, key=lambda x: x[2])
    for i, (name, params, flops) in enumerate(models_by_flops, 1):
        print(f"   {i}. {name}: {flops:,} FLOPs ({flops/1e9:.3f} GFLOPs)")
    
    print("\n3. 效率分析:")
    print("   - RNN_lstm: 参数量最少，计算量适中，适合实时应用")
    print("   - simple_CNN: 中等复杂度，在参数量和性能间取得平衡")
    print("   - TwobythreeR_CNN: 最复杂的模型，多尺度特征提取，可能有最好的性能但计算成本最高")

if __name__ == "__main__":
    main()