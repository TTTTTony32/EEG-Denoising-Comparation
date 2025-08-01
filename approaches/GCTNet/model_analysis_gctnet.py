"""
GCTNet (Global Context Transformer Network) 模型参数量和FLOPs分析
结合CNN和Transformer的混合架构EEG去噪模型
"""

def analyze_MultiHeadAttention(emb_size, num_heads, seq_len):
    """
    分析MultiHeadAttention模块
    """
    # 参数量
    # keys, queries, values: 每个都是 Linear(emb_size, emb_size)
    keys_params = emb_size * emb_size + emb_size
    queries_params = emb_size * emb_size + emb_size
    values_params = emb_size * emb_size + emb_size
    # projection: Linear(emb_size, emb_size)
    projection_params = emb_size * emb_size + emb_size
    
    total_params = keys_params + queries_params + values_params + projection_params
    
    # FLOPs
    # Q, K, V计算: 3 × (seq_len × emb_size × emb_size)
    qkv_flops = 3 * (seq_len * emb_size * emb_size)
    
    # Q @ K^T: num_heads × seq_len × (emb_size/num_heads) × seq_len
    head_dim = emb_size // num_heads
    qk_flops = num_heads * seq_len * head_dim * seq_len
    
    # Softmax: num_heads × seq_len × seq_len
    softmax_flops = num_heads * seq_len * seq_len
    
    # Attention @ V: num_heads × seq_len × seq_len × head_dim
    av_flops = num_heads * seq_len * seq_len * head_dim
    
    # Output projection: seq_len × emb_size × emb_size
    proj_flops = seq_len * emb_size * emb_size
    
    total_flops = qkv_flops + qk_flops + softmax_flops + av_flops + proj_flops
    
    return total_params, total_flops

def analyze_FeedForwardBlock(emb_size, expansion, seq_len):
    """
    分析FeedForwardBlock模块
    """
    hidden_size = expansion * emb_size
    
    # 参数量
    # Linear1: emb_size → hidden_size
    linear1_params = emb_size * hidden_size + hidden_size
    # Linear2: hidden_size → emb_size
    linear2_params = hidden_size * emb_size + emb_size
    total_params = linear1_params + linear2_params
    
    # FLOPs
    # Linear1: seq_len × emb_size × hidden_size
    linear1_flops = seq_len * emb_size * hidden_size
    # Swish activation: seq_len × hidden_size × 2 (sigmoid + multiply)
    swish_flops = seq_len * hidden_size * 2
    # Linear2: seq_len × hidden_size × emb_size
    linear2_flops = seq_len * hidden_size * emb_size
    total_flops = linear1_flops + swish_flops + linear2_flops
    
    return total_params, total_flops

def analyze_TransformerEncoderBlock(emb_size, num_heads, seq_len, forward_expansion=1):
    """
    分析TransformerEncoderBlock模块
    """
    # MultiHeadAttention
    mha_params, mha_flops = analyze_MultiHeadAttention(emb_size, num_heads, seq_len)
    
    # FeedForwardBlock
    ffn_params, ffn_flops = analyze_FeedForwardBlock(emb_size, forward_expansion, seq_len)
    
    # LayerNorm × 2 (每个都有 2 × emb_size 个参数)
    ln_params = 2 * (2 * emb_size)
    # LayerNorm FLOPs (近似): 2 × (4 × emb_size × seq_len)
    ln_flops = 2 * (4 * emb_size * seq_len)
    
    total_params = mha_params + ffn_params + ln_params
    total_flops = mha_flops + ffn_flops + ln_flops
    
    return total_params, total_flops

def analyze_CNN_block(in_channels, out_channels, kernel_size, seq_len, has_pool=True, pool_factor=2):
    """
    分析CNN block (Conv1d + BatchNorm + Activation)
    """
    # 参数量
    # Conv1d: in_channels × out_channels × kernel_size + out_channels(bias)
    conv_params = in_channels * out_channels * kernel_size + out_channels
    # BatchNorm1d: 2 × out_channels (scale + shift)
    bn_params = 2 * out_channels
    total_params = conv_params + bn_params
    
    # FLOPs
    # Conv1d: seq_len × in_channels × out_channels × kernel_size
    conv_flops = seq_len * in_channels * out_channels * kernel_size
    # BatchNorm1d: 4 × out_channels × seq_len (近似)
    bn_flops = 4 * out_channels * seq_len
    # LeakyReLU: out_channels × seq_len
    relu_flops = out_channels * seq_len
    total_flops = conv_flops + bn_flops + relu_flops
    
    # 更新序列长度 (如果有pooling)
    if has_pool:
        new_seq_len = seq_len // pool_factor
    else:
        new_seq_len = seq_len
    
    return total_params, total_flops, new_seq_len

def analyze_Generator(data_num=512):
    """
    分析主要的Generator模型 (GCTNet)
    """
    print("="*70)
    print("GCTNET - Generator (Global Context Transformer) 模型分析")
    print("="*70)
    print(f"输入维度: (batch_size, 1, {data_num})")
    
    total_params = 0
    total_flops = 0
    seq_len = data_num
    
    # 1. Block1 (初始CNN层)
    print("\n--- Block1 (初始特征提取) ---")
    # Conv1d(1, 32, 3) + Conv1d(32, 32, 3) + AvgPool1d(2)
    conv1_params, conv1_flops, seq_len = analyze_CNN_block(1, 32, 3, seq_len)
    conv2_params, conv2_flops, seq_len = analyze_CNN_block(32, 32, 3, seq_len, has_pool=True)
    
    block1_params = conv1_params + conv2_params
    block1_flops = conv1_flops + conv2_flops
    total_params += block1_params
    total_flops += block1_flops
    print(f"Block1: 参数={block1_params:,}, FLOPs={block1_flops:,}, 输出长度={seq_len}")
    
    # 2-6. 混合CNN+Transformer blocks
    block_configs = [
        {"in_ch": 32, "out_ch": 64, "emb_size": 32},     # block2
        {"in_ch": 64, "out_ch": 128, "emb_size": 64},    # block3  
        {"in_ch": 128, "out_ch": 256, "emb_size": 128},  # block4
        {"in_ch": 256, "out_ch": 512, "emb_size": 256},  # block5
        {"in_ch": 512, "out_ch": 1024, "emb_size": 512}, # block6
    ]
    
    for i, config in enumerate(block_configs, 2):
        print(f"\n--- Block{i} (CNN + Transformer 混合) ---")
        
        # CNN分支: Conv1d + Conv1d + AvgPool1d
        cnn_conv1_params, cnn_conv1_flops, _ = analyze_CNN_block(
            config["in_ch"], config["out_ch"], 3, seq_len, has_pool=False
        )
        cnn_conv2_params, cnn_conv2_flops, cnn_seq_len = analyze_CNN_block(
            config["out_ch"], config["out_ch"], 3, seq_len, has_pool=True
        )
        cnn_branch_params = cnn_conv1_params + cnn_conv2_params
        cnn_branch_flops = cnn_conv1_flops + cnn_conv2_flops
        
        # Transformer分支: TransformerEncoderBlock + Conv1d(stride=2)
        trans_params, trans_flops = analyze_TransformerEncoderBlock(
            config["emb_size"], num_heads=8, seq_len=seq_len
        )
        trans_conv_params, trans_conv_flops, trans_seq_len = analyze_CNN_block(
            config["emb_size"], config["out_ch"], 3, seq_len, has_pool=False
        )
        # stride=2相当于pooling
        trans_seq_len = trans_seq_len // 2
        trans_branch_params = trans_params + trans_conv_params
        trans_branch_flops = trans_flops + trans_conv_flops
        
        # 特征融合模块 FFM: Conv1d(2*out_ch, out_ch, 3)
        ffm_params, ffm_flops, _ = analyze_CNN_block(
            2 * config["out_ch"], config["out_ch"], 3, cnn_seq_len, has_pool=False
        )
        
        block_params = cnn_branch_params + trans_branch_params + ffm_params
        block_flops = cnn_branch_flops + trans_branch_flops + ffm_flops
        
        total_params += block_params
        total_flops += block_flops
        seq_len = cnn_seq_len  # 更新序列长度
        
        print(f"  CNN分支: 参数={cnn_branch_params:,}, FLOPs={cnn_branch_flops:,}")
        print(f"  Transformer分支: 参数={trans_branch_params:,}, FLOPs={trans_branch_flops:,}")
        print(f"  特征融合FFM: 参数={ffm_params:,}, FLOPs={ffm_flops:,}")
        print(f"  Block{i}总计: 参数={block_params:,}, FLOPs={block_flops:,}, 输出长度={seq_len}")
    
    # 7. Block7 (最终处理)
    print("\n--- Block7 (最终处理) ---")
    conv7_1_params, conv7_1_flops, seq_len = analyze_CNN_block(1024, 1024, 3, seq_len, has_pool=False)
    conv7_2_params, conv7_2_flops, seq_len = analyze_CNN_block(1024, 1024, 3, seq_len, has_pool=False)
    
    block7_params = conv7_1_params + conv7_2_params
    block7_flops = conv7_1_flops + conv7_2_flops
    total_params += block7_params
    total_flops += block7_flops
    print(f"Block7: 参数={block7_params:,}, FLOPs={block7_flops:,}")
    
    # 8. 最终线性层
    print("\n--- 最终线性层 ---")
    # 根据架构，最终特征维度应该是 1024 × seq_len
    # 但代码中写的是 16 * 512，这可能是硬编码的
    final_features = 16 * 512  # 从代码中的硬编码值
    linear_params = final_features * data_num + data_num
    linear_flops = final_features * data_num
    
    total_params += linear_params
    total_flops += linear_flops
    print(f"Linear层: 参数={linear_params:,}, FLOPs={linear_flops:,}")
    
    print(f"\n总参数量: {total_params:,}")
    print(f"总FLOPs: {total_flops:,}")
    print(f"总FLOPs (MFLOPs): {total_flops/1e6:.3f}")
    print(f"总FLOPs (GFLOPs): {total_flops/1e9:.3f}")
    
    return total_params, total_flops

def analyze_Discriminator():
    """
    分析Discriminator模型
    """
    print("\n" + "="*70)
    print("GCTNET - Discriminator 模型分析")
    print("="*70)
    
    total_params = 0
    total_flops = 0
    seq_len = 512  # 输入长度
    
    # Conv layers分析
    conv_configs = [
        # conv1
        [(1, 64, 3, 2), (64, 64, 3, 2)],
        # conv2  
        [(64, 128, 3, 2), (128, 128, 3, 2)],
        # conv3
        [(128, 256, 3, 2), (256, 256, 3, 2)],
        # model (最终卷积层)
        [(256, 512, 3, 2), (512, 512, 3, 2)],
    ]
    
    for i, block_config in enumerate(conv_configs, 1):
        block_params = 0
        block_flops = 0
        
        for in_ch, out_ch, kernel, stride in block_config:
            # Conv1d参数: in_ch × out_ch × kernel + out_ch
            conv_params = in_ch * out_ch * kernel + out_ch
            # BatchNorm1d参数: 2 × out_ch
            bn_params = 2 * out_ch
            layer_params = conv_params + bn_params
            
            # Conv1d FLOPs: seq_len × in_ch × out_ch × kernel
            conv_flops = seq_len * in_ch * out_ch * kernel
            # BatchNorm + LeakyReLU FLOPs
            other_flops = 5 * out_ch * seq_len
            layer_flops = conv_flops + other_flops
            
            block_params += layer_params
            block_flops += layer_flops
            seq_len = seq_len // stride  # 更新序列长度
        
        total_params += block_params
        total_flops += block_flops
        print(f"Block{i}: 参数={block_params:,}, FLOPs={block_flops:,}, 输出长度={seq_len}")
    
    # 最终全连接层
    # 假设最终特征维度为 512 × 2 = 1024 (从代码推测)
    final_features = 1024
    dense_params = final_features * 1 + 1  # 输出1维(真假判别)
    dense_flops = final_features * 1
    
    total_params += dense_params
    total_flops += dense_flops
    print(f"Dense层: 参数={dense_params:,}, FLOPs={dense_flops:,}")
    
    print(f"\nDiscriminator总参数量: {total_params:,}")
    print(f"Discriminator总FLOPs: {total_flops:,}")
    print(f"Discriminator总FLOPs (MFLOPs): {total_flops/1e6:.3f}")
    
    return total_params, total_flops

def compare_generator_variants():
    """
    比较不同的Generator变体
    """
    print("\n" + "="*80)
    print("GCTNet Generator变体对比")
    print("="*80)
    
    print("GCTNet包含三种Generator变体:")
    print("1. GeneratorCNN: 纯CNN架构")
    print("2. GeneratorTransformer: 纯Transformer架构")  
    print("3. Generator (GCT): CNN + Transformer混合架构")
    
    # 这里只分析主要的Generator (混合架构)
    # 其他变体的详细分析可以类似进行
    
    print(f"\n主要分析: Generator (GCT) - 混合架构")
    print("- 每个block包含CNN分支和Transformer分支")
    print("- 使用特征融合模块(FFM)合并两个分支")
    print("- Transformer分支使用标准的Multi-Head Attention")
    print("- CNN分支使用传统卷积+池化")

def main():
    """主函数：分析GCTNet模型"""
    data_num = 512  # EEG数据长度
    
    print("GCTNet (Global Context Transformer Network) 模型分析")
    print(f"输入数据长度: {data_num}")
    
    # 分析Generator
    gen_params, gen_flops = analyze_Generator(data_num)
    
    # 分析Discriminator  
    disc_params, disc_flops = analyze_Discriminator()
    
    # 变体对比
    compare_generator_variants()
    
    # 总体分析
    print("\n" + "="*80)
    print("GCTNet模型总体分析")
    print("="*80)
    
    total_params = gen_params + disc_params
    total_flops = gen_flops + disc_flops
    
    print(f"{'组件':<20} {'参数量':<15} {'FLOPs(M)':<12}")
    print("-" * 50)
    print(f"{'Generator':<20} {gen_params:<15,} {gen_flops/1e6:<12.1f}")
    print(f"{'Discriminator':<20} {disc_params:<15,} {disc_flops/1e6:<12.1f}")
    print(f"{'总计':<20} {total_params:<15,} {total_flops/1e6:<12.1f}")
    
    # 技术特点分析
    print("\n" + "="*80)
    print("GCTNet技术特点分析")
    print("="*80)
    
    print("1. 混合架构设计:")
    print("   - CNN分支: 局部特征提取，计算效率高")
    print("   - Transformer分支: 全局上下文建模，长程依赖")
    print("   - 特征融合: FFM模块融合两个分支的优势")
    
    print("\n2. 多尺度处理:")
    print("   - 6个层次的渐进式特征提取")
    print("   - 每层特征图尺寸减半，通道数翻倍")
    print("   - 适应不同尺度的EEG模式")
    
    print("\n3. 对抗训练:")
    print("   - Generator: 生成干净的EEG信号")
    print("   - Discriminator: 区分真实和生成的信号")
    print("   - 提高去噪质量和信号真实性")
    
    print("\n4. 计算特点:")
    print("   - 参数量较大，适合高性能设备")
    print("   - 结合CNN效率和Transformer表达能力")
    print("   - 对抗训练需要额外计算开销")
    
    print("\n5. 适用场景:")
    print("   - 高质量EEG去噪任务")
    print("   - 需要保持信号细节和全局一致性")
    print("   - 有充足计算资源的应用场景")

if __name__ == "__main__":
    main()