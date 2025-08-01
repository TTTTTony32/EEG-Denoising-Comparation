"""
IFN (Interactive Fusion Network) 模型参数量和FLOPs分析
基于双分支交互融合架构的EEG去噪模型
"""

def analyze_Conv_Block(channel, kernel, seq_len):
    """
    分析Conv_Block模块
    """
    params = 0
    flops = 0
    
    # lay1: Conv1d(channel, channel//2, kernel) + BatchNorm + Dropout + Sigmoid
    conv1_params = channel * (channel // 2) * kernel  # no bias
    bn1_params = 2 * (channel // 2)  # scale + bias
    
    # lay2: 4个Conv1d + 3个BatchNorm
    conv2_1_params = channel * channel * kernel + channel
    conv2_2_params = channel * channel * kernel + channel  
    conv2_3_params = channel * (channel // 2) * kernel  # no bias in final layer
    bn2_params = 3 * (2 * channel) + 0  # 只有前两个有BN
    
    total_params = conv1_params + bn1_params + conv2_1_params + conv2_2_params + conv2_3_params + bn2_params
    
    # FLOPs计算
    conv1_flops = channel * (channel // 2) * kernel * seq_len
    conv2_1_flops = channel * channel * kernel * seq_len
    conv2_2_flops = channel * channel * kernel * seq_len
    conv2_3_flops = channel * (channel // 2) * kernel * seq_len
    
    # BatchNorm + 激活函数
    bn_flops = seq_len * ((channel // 2) + 3 * channel) * 4  # 近似
    activation_flops = seq_len * (channel // 2 + 3 * channel) * 2  # ReLU + Sigmoid
    
    total_flops = conv1_flops + conv2_1_flops + conv2_2_flops + conv2_3_flops + bn_flops + activation_flops
    
    return total_params, total_flops

def analyze_Interaction_Block(channel, outchannel, seq_len):
    """
    分析Interaction_Block模块 (双分支交互)
    """
    params = 0
    flops = 0
    
    # Conv_n2s 和 Conv_s2n (两个Conv_Block，输入通道为channel*2)
    conv_block_params, conv_block_flops = analyze_Conv_Block(channel * 2, 9, seq_len)
    params += conv_block_params * 2  # 两个Conv_Block
    flops += conv_block_flops * 2
    
    # lay_s 和 lay_n: Conv1d(channel, outchannel, 9) + BatchNorm + ReLU + Dropout
    lay_params = 2 * (channel * outchannel * 9 + outchannel + 2 * outchannel)  # Conv + BN
    lay_flops = 2 * (channel * outchannel * 9 * seq_len + 4 * outchannel * seq_len)  # Conv + BN + ReLU
    
    params += lay_params
    flops += lay_flops
    
    # Concatenation和element-wise operations
    concat_flops = 2 * seq_len * channel  # cat操作
    elementwise_flops = 2 * seq_len * channel  # element-wise multiplication
    
    flops += concat_flops + elementwise_flops
    
    return params, flops

def analyze_GRU(input_size, hidden_size, seq_len, bidirectional=True):
    """
    分析GRU模块
    """
    # GRU参数: 3个门 (reset, update, new) × (input_size + hidden_size + 1) × hidden_size
    gru_params = 3 * (input_size + hidden_size + 1) * hidden_size
    
    if bidirectional:
        gru_params *= 2
    
    # GRU FLOPs: 3 gates × (input + hidden) × hidden × seq_len
    gru_flops = 3 * (input_size + hidden_size) * hidden_size * seq_len
    if bidirectional:
        gru_flops *= 2
    
    return gru_params, gru_flops

def analyze_OA_INet(seq_len=512):
    """
    分析OA_INet模型 (无下采样版本)
    """
    print("="*70)
    print("IFN - OA_INet (双分支交互网络 - 无下采样) 分析")
    print("="*70)
    print(f"输入维度: (batch_size, 1, {seq_len})")
    
    params = 0
    flops = 0
    
    # 1. 初始卷积层 (双分支)
    print("\n--- 初始卷积层 (双分支) ---")
    # c1_e 和 c1_n: Conv1d(1, 32, 9) + BatchNorm + ReLU + Dropout
    init_conv_params = 2 * (1 * 32 * 9 + 32 + 2 * 32)  # Conv + BN
    init_conv_flops = 2 * (1 * 32 * 9 * seq_len + 4 * 32 * seq_len)  # Conv + BN + ReLU
    
    params += init_conv_params
    flops += init_conv_flops
    print(f"初始卷积: 参数={init_conv_params:,}, FLOPs={init_conv_flops:,}")
    
    # 2. 第一个交互模块
    print("\n--- 交互模块1 ---")
    i1_params, i1_flops = analyze_Interaction_Block(32, 8, seq_len)
    params += i1_params
    flops += i1_flops
    print(f"Interaction_Block1: 参数={i1_params:,}, FLOPs={i1_flops:,}")
    
    # 3. 第二层卷积 (双分支，输入通道32+8=40)
    print("\n--- 第二层卷积 (双分支) ---")
    conv2_params = 2 * (40 * 32 * 9 + 32 + 2 * 32)  # Conv + BN
    conv2_flops = 2 * (40 * 32 * 9 * seq_len + 4 * 32 * seq_len)
    
    params += conv2_params
    flops += conv2_flops
    print(f"第二层卷积: 参数={conv2_params:,}, FLOPs={conv2_flops:,}")
    
    # 4. 第二个交互模块
    print("\n--- 交互模块2 ---")
    i2_params, i2_flops = analyze_Interaction_Block(32, 8, seq_len)
    params += i2_params
    flops += i2_flops
    print(f"Interaction_Block2: 参数={i2_params:,}, FLOPs={i2_flops:,}")
    
    # 5. 第三层卷积 (双分支)
    print("\n--- 第三层卷积 (双分支) ---")
    conv3_params = conv2_params  # 相同结构
    conv3_flops = conv2_flops
    
    params += conv3_params
    flops += conv3_flops
    print(f"第三层卷积: 参数={conv3_params:,}, FLOPs={conv3_flops:,}")
    
    # 6. 第三个交互模块
    print("\n--- 交互模块3 ---")
    i3_params, i3_flops = analyze_Interaction_Block(32, 8, seq_len)
    params += i3_params
    flops += i3_flops
    print(f"Interaction_Block3: 参数={i3_params:,}, FLOPs={i3_flops:,}")
    
    # 7. GRU层 (双分支)
    print("\n--- GRU层 (双分支) ---")
    gru_params, gru_flops = analyze_GRU(40, 32, seq_len, bidirectional=True)  # 输入维度40
    params += gru_params * 2  # 两个GRU分支
    flops += gru_flops * 2
    print(f"双向GRU (x2): 参数={gru_params*2:,}, FLOPs={gru_flops*2:,}")
    
    # 8. 全连接层
    print("\n--- 全连接层 ---")
    # f1_e, f1_n: Linear(64*512, 512)
    f1_params = 2 * (64 * seq_len * 512 + 512)
    f1_flops = 2 * (64 * seq_len * 512)
    
    # fc2, fc3, fc4, fc5: Linear(512, 512)
    fc_params = 4 * (512 * 512 + 512)
    fc_flops = 4 * (512 * 512)
    
    total_fc_params = f1_params + fc_params
    total_fc_flops = f1_flops + fc_flops
    
    params += total_fc_params
    flops += total_fc_flops
    print(f"全连接层: 参数={total_fc_params:,}, FLOPs={total_fc_flops:,}")
    
    print(f"\n总参数量: {params:,}")
    print(f"总FLOPs: {flops:,}")
    print(f"总FLOPs (MFLOPs): {flops/1e6:.3f}")
    print(f"总FLOPs (GFLOPs): {flops/1e9:.3f}")
    
    return params, flops

def analyze_MA_INet(seq_len=512):
    """
    分析MA_INet模型 (下采样版本)
    """
    print("\n" + "="*70)
    print("IFN - MA_INet (双分支交互网络 - 下采样) 分析")
    print("="*70)
    print(f"输入维度: (batch_size, 1, {seq_len})")
    
    params = 0
    flops = 0
    
    # 计算下采样后的序列长度
    # stride=2, 三次下采样: 512 -> 256 -> 128 -> 64
    seq_len_after_downsample = seq_len // (2**3)  # 64
    
    print(f"下采样后序列长度: {seq_len_after_downsample}")
    
    # 1. 三层下采样卷积 (每层stride=2)
    print("\n--- 下采样卷积层 ---")
    current_seq_len = seq_len
    
    # Layer 1: Conv1d(1, 32, 9, stride=2)
    conv1_params = 2 * (1 * 32 * 9 + 32 + 2 * 32)
    current_seq_len = current_seq_len // 2  # 256
    conv1_flops = 2 * (1 * 32 * 9 * current_seq_len + 4 * 32 * current_seq_len)
    
    # Layer 2: Conv1d(32, 32, 9, stride=2)  
    conv2_params = 2 * (32 * 32 * 9 + 32 + 2 * 32)
    current_seq_len = current_seq_len // 2  # 128
    conv2_flops = 2 * (32 * 32 * 9 * current_seq_len + 4 * 32 * current_seq_len)
    
    # Layer 3: Conv1d(32, 32, 9, stride=2)
    conv3_params = conv2_params
    current_seq_len = current_seq_len // 2  # 64
    conv3_flops = 2 * (32 * 32 * 9 * current_seq_len + 4 * 32 * current_seq_len)
    
    downsample_params = conv1_params + conv2_params + conv3_params
    downsample_flops = conv1_flops + conv2_flops + conv3_flops
    
    params += downsample_params
    flops += downsample_flops
    print(f"下采样卷积: 参数={downsample_params:,}, FLOPs={downsample_flops:,}")
    
    # 2. GRU层 (双分支，输入维度32，序列长度64)
    print("\n--- GRU层 (双分支) ---")
    gru_params, gru_flops = analyze_GRU(32, 32, seq_len_after_downsample, bidirectional=True)
    params += gru_params * 2
    flops += gru_flops * 2
    print(f"双向GRU (x2): 参数={gru_params*2:,}, FLOPs={gru_flops*2:,}")
    
    # 3. 全连接层 (序列长度变小)
    print("\n--- 全连接层 ---")
    # f1_e, f1_n: Linear(64*64, 512) - 注意这里序列长度是64
    f1_params = 2 * (64 * seq_len_after_downsample * 512 + 512)
    f1_flops = 2 * (64 * seq_len_after_downsample * 512)
    
    # fc2, fc4: Linear(512, 512) - 只有e分支有后续fc层
    fc_params = 2 * (512 * 512 + 512)
    fc_flops = 2 * (512 * 512)
    
    total_fc_params = f1_params + fc_params
    total_fc_flops = f1_flops + fc_flops
    
    params += total_fc_params
    flops += total_fc_flops
    print(f"全连接层: 参数={total_fc_params:,}, FLOPs={total_fc_flops:,}")
    
    print(f"\n总参数量: {params:,}")
    print(f"总FLOPs: {flops:,}")
    print(f"总FLOPs (MFLOPs): {flops/1e6:.3f}")
    print(f"总FLOPs (GFLOPs): {flops/1e9:.3f}")
    
    return params, flops

def analyze_fusion_mechanisms():
    """
    分析IFN的融合机制特点
    """
    print("\n" + "="*80)
    print("IFN 双分支融合机制分析")
    print("="*80)
    
    print("1. 核心设计理念:")
    print("   - 双分支架构: 分别建模干净信号和噪声特征")
    print("   - 交互式融合: 通过Interaction_Block实现分支间信息交换")
    print("   - 多尺度处理: 不同层次的特征交互")
    print("   - 时序建模: GRU捕获时间依赖关系")
    
    print("\n2. Interaction_Block 工作原理:")
    print("   - 输入: F_RA_s (信号分支), F_RA_n (噪声分支)")
    print("   - Concatenation: 将两分支特征连接")
    print("   - Mask Generation: 生成交互掩码")
    print("   - Cross-Enhancement: H_n2s = F_RA_n * Mask_n")
    print("   - Feature Fusion: 结合原始特征和交互特征")
    
    print("\n3. Exchange机制 (可选):")
    print("   - 基于BatchNorm权重的特征交换")
    print("   - 选择最小权重的K个通道进行交换")
    print("   - 增强分支间的特征互补性")
    
    print("\n4. 两种网络变体:")
    print("   - OA_INet: 保持原始分辨率，完整特征交互")
    print("   - MA_INet: 下采样减少计算量，适合实时应用")
    
    print("\n5. 掩码网络 (MA_MNet/OA_MNet):")
    print("   - 自适应融合机制: mask = sigmoid(conv(concat(x, x1, x2)))")
    print("   - 软性选择: out = x1*mask + x2*(1-mask)")
    print("   - 学习最优融合权重")

def compare_ifn_variants():
    """
    比较IFN的不同变体
    """
    print("\n" + "="*80)
    print("IFN变体对比分析")
    print("="*80)
    
    # 分析默认配置
    seq_len = 512
    oa_params, oa_flops = analyze_OA_INet(seq_len)
    ma_params, ma_flops = analyze_MA_INet(seq_len)
    
    print(f"\n{'模型变体':<15} {'参数量':<15} {'FLOPs(M)':<12} {'特点':<30}")
    print("-" * 80)
    print(f"{'OA_INet':<15} {oa_params:<15,} {oa_flops/1e6:<12.1f} {'保持分辨率,完整交互':<30}")
    print(f"{'MA_INet':<15} {ma_params:<15,} {ma_flops/1e6:<12.1f} {'下采样,高效计算':<30}")
    
    print(f"\n计算效率对比:")
    flops_reduction = oa_flops / ma_flops
    params_reduction = oa_params / ma_params
    print(f"MA_INet相比OA_INet:")
    print(f"- FLOPs减少: {flops_reduction:.1f}x")
    print(f"- 参数减少: {params_reduction:.1f}x")
    
    print(f"\n设计权衡:")
    print(f"- OA_INet: 高精度，适合离线处理")
    print(f"- MA_INet: 高效率，适合实时应用")
    print(f"- 两者都保持双分支交互的核心设计")

def main():
    """主函数：分析IFN模型系列"""
    seq_len = 512  # 标准EEG序列长度
    
    print("IFN (Interactive Fusion Network) 双分支融合模型分析")
    print(f"输入序列长度: {seq_len}")
    
    # 分析两个主要变体
    oa_params, oa_flops = analyze_OA_INet(seq_len)
    ma_params, ma_flops = analyze_MA_INet(seq_len)
    
    # 分析融合机制
    analyze_fusion_mechanisms()
    
    # 比较不同变体
    compare_ifn_variants()
    
    # 技术特点总结
    print("\n" + "="*80)
    print("IFN 技术特点总结")
    print("="*80)
    
    print("1. 创新点:")
    print("   - 双分支交互式融合架构")
    print("   - 多层次特征交互机制")
    print("   - 自适应掩码融合策略")
    print("   - 时序-空间联合建模")
    
    print("\n2. 技术优势:")
    print("   - 显式建模信号-噪声交互")
    print("   - 灵活的特征交换机制")
    print("   - 多尺度时序建模能力")
    print("   - 端到端优化的融合权重")
    
    print("\n3. 适用场景:")
    print("   - 复杂噪声环境的EEG去噪")
    print("   - 需要精细特征交互的任务")
    print("   - 信号质量要求较高的应用")
    print("   - 多模态信息融合场景")
    
    print("\n4. 计算特性:")
    print(f"   - OA_INet: {oa_params:,}参数, {oa_flops/1e6:.1f}M FLOPs")
    print(f"   - MA_INet: {ma_params:,}参数, {ma_flops/1e6:.1f}M FLOPs")
    print("   - 双分支并行计算友好")
    print("   - 可选的实时优化版本")

if __name__ == "__main__":
    main()