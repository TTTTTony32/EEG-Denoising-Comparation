"""
EEGDfus模型参数量和FLOPs分析
包含两个模型变体：EEGDNet和SSED
"""

def analyze_DualBranchDenoisingModel_EEGDNet(datanum=512, feats=64):
    """
    分析EEGDNet版本的DualBranchDenoisingModel
    基于denoising_model_eegdnet.py
    """
    print("="*70)
    print("EEGDFUS - DUALBRANCH DENOISING MODEL (EEGDNet版本) 分析")
    print("="*70)
    print(f"输入维度: (batch_size, 1, {datanum})")
    print(f"特征维度: {feats}")
    
    # 模型超参数 (from denoising_model_eegdnet.py)
    d_model = 512  # Embedding Size
    d_ff = 512     # FeedForward dimension
    d_k = d_v = 64 # dimension of K(=Q), V
    n_heads = 1    # number of heads in Multi-Head Attention
    
    print(f"Transformer参数: d_model={d_model}, d_ff={d_ff}, d_k={d_k}, n_heads={n_heads}")
    
    params = 0
    flops = 0
    
    # 1. Stream_x 分支
    print("\n--- Stream_x 分支 ---")
    
    # Conv1D层 (1->feats, feats->feats)
    conv1_params = 1 * feats * 3 + feats  # Conv1d(1, feats, 3)
    conv2_params = feats * feats * 3 + feats  # Conv1d(feats, feats, 3)
    conv_params_x = conv1_params + conv2_params
    
    conv1_flops = 1 * feats * 3 * datanum
    conv2_flops = feats * feats * 3 * datanum  
    conv_flops_x = conv1_flops + conv2_flops
    
    params += conv_params_x
    flops += conv_flops_x
    print(f"Conv1D层: 参数={conv_params_x:,}, FLOPs={conv_flops_x:,}")
    
    # EncoderLayer x3 (MultiHeadAttention + PoswiseFeedForwardNet)
    # MultiHeadAttention参数:
    # W_Q: d_model * (d_k * n_heads) = 512 * 64 = 32,768
    # W_K: d_model * (d_k * n_heads) = 512 * 64 = 32,768  
    # W_V: d_model * (d_v * n_heads) = 512 * 64 = 32,768
    # fc: (n_heads * d_v) * d_model = 64 * 512 = 32,768
    # LayerNorm: 2 * d_model = 1,024
    mha_params = (d_model * d_k * n_heads) * 3 + (n_heads * d_v * d_model) + (2 * d_model)
    
    # PoswiseFeedForwardNet参数:
    # fc1: d_model * d_ff = 512 * 512 = 262,144
    # fc2: d_ff * d_model = 512 * 512 = 262,144  
    # LayerNorm: 2 * d_model = 1,024
    ffn_params = (d_model * d_ff) + (d_ff * d_model) + (2 * d_model)
    
    encoder_layer_params = mha_params + ffn_params
    encoder_layers_params_x = encoder_layer_params * 3  # 3个EncoderLayer
    
    params += encoder_layers_params_x
    print(f"3个EncoderLayer: 参数={encoder_layers_params_x:,}")
    
    # EncoderLayer FLOPs计算
    # MultiHeadAttention FLOPs:
    # Q*K^T: d_k * datanum * datanum
    # softmax: datanum * datanum  
    # attention*V: datanum * datanum * d_v
    # 线性变换: datanum * (d_model * d_k * 3 + d_v * d_model)
    mha_flops = (d_k * datanum * datanum) + (datanum * datanum) + (datanum * datanum * d_v) + \
                (datanum * (d_model * d_k * 3 + d_v * d_model))
    
    # FFN FLOPs: 
    # fc1: datanum * d_model * d_ff
    # fc2: datanum * d_ff * d_model  
    ffn_flops = datanum * d_model * d_ff * 2
    
    encoder_layer_flops = mha_flops + ffn_flops
    encoder_layers_flops_x = encoder_layer_flops * 3
    flops += encoder_layers_flops_x
    print(f"3个EncoderLayer FLOPs: {encoder_layers_flops_x:,}")
    
    # 2. Stream_cond 分支 (相同结构)
    print("\n--- Stream_cond 分支 ---")
    conv_params_cond = conv_params_x
    conv_flops_cond = conv_flops_x
    encoder_layers_params_cond = encoder_layers_params_x
    encoder_layers_flops_cond = encoder_layers_flops_x
    
    params += conv_params_cond + encoder_layers_params_cond
    flops += conv_flops_cond + encoder_layers_flops_cond
    print(f"Conv1D层: 参数={conv_params_cond:,}, FLOPs={conv_flops_cond:,}")
    print(f"3个EncoderLayer: 参数={encoder_layers_params_cond:,}, FLOPs={encoder_layers_flops_cond:,}")
    
    # 3. PositionalEncoding (embed)
    print("\n--- PositionalEncoding ---")
    # 没有可训练参数，只有计算
    pos_enc_flops = feats * 2  # sin + cos计算
    flops += pos_enc_flops
    print(f"PositionalEncoding FLOPs: {pos_enc_flops}")
    
    # 4. Bridge (FiLM layers x4)
    print("\n--- Bridge (FiLM layers) ---")
    # 每个FiLM层:
    # fc_gamma: condition_dim * input_dim = 1 * d_model = 512
    # fc_beta: condition_dim * input_dim = 1 * d_model = 512  
    film_params_per_layer = (1 * d_model) * 2  # gamma + beta
    film_params_total = film_params_per_layer * 4  # 4个FiLM层
    
    # FiLM FLOPs: 2 * d_model * datanum (gamma*x + beta)
    film_flops_per_layer = 2 * d_model * datanum
    film_flops_total = film_flops_per_layer * 4
    
    params += film_params_total
    flops += film_flops_total
    print(f"4个FiLM层: 参数={film_params_total:,}, FLOPs={film_flops_total:,}")
    
    # 5. 输出卷积层
    print("\n--- 输出卷积层 ---")
    # Conv1d(feats, feats, 3) + Conv1d(feats, 1, 3)
    conv_out1_params = feats * feats * 3 + feats
    conv_out2_params = feats * 1 * 3 + 1
    conv_out_params = conv_out1_params + conv_out2_params
    
    conv_out1_flops = feats * feats * 3 * datanum
    conv_out2_flops = feats * 1 * 3 * datanum
    conv_out_flops = conv_out1_flops + conv_out2_flops
    
    params += conv_out_params
    flops += conv_out_flops
    print(f"输出Conv层: 参数={conv_out_params:,}, FLOPs={conv_out_flops:,}")
    
    print(f"\n总参数量: {params:,}")
    print(f"总FLOPs: {flops:,}")
    print(f"总FLOPs (GFLOPs): {flops/1e9:.3f}")
    
    return params, flops

def analyze_DualBranchDenoisingModel_SSED(datanum=512, feats=64):
    """
    分析SSED版本的DualBranchDenoisingModel  
    基于denoising_model_seed.py (参数稍有不同)
    """
    print("\n" + "="*70)
    print("EEGDFUS - DUALBRANCH DENOISING MODEL (SSED版本) 分析")
    print("="*70)
    print(f"输入维度: (batch_size, 1, {datanum})")
    print(f"特征维度: {feats}")
    
    # 模型超参数 (from denoising_model_seed.py)
    d_model = 400  # Embedding Size (不同)
    d_ff = 512     # FeedForward dimension
    d_k = d_v = 32 # dimension of K(=Q), V (不同)
    n_heads = 1    # number of heads in Multi-Head Attention
    
    print(f"Transformer参数: d_model={d_model}, d_ff={d_ff}, d_k={d_k}, n_heads={n_heads}")
    
    params = 0
    flops = 0
    
    # 计算逻辑与EEGDNet版本相同，但参数不同
    # 1. Stream_x 分支
    print("\n--- Stream_x 分支 ---")
    
    # Conv1D层
    conv1_params = 1 * feats * 3 + feats
    conv2_params = feats * feats * 3 + feats
    conv_params_x = conv1_params + conv2_params
    
    conv1_flops = 1 * feats * 3 * datanum
    conv2_flops = feats * feats * 3 * datanum
    conv_flops_x = conv1_flops + conv2_flops
    
    params += conv_params_x
    flops += conv_flops_x
    print(f"Conv1D层: 参数={conv_params_x:,}, FLOPs={conv_flops_x:,}")
    
    # EncoderLayer x3 (参数调整为d_model=400, d_k=32)
    mha_params = (d_model * d_k * n_heads) * 3 + (n_heads * d_v * d_model) + (2 * d_model)
    ffn_params = (d_model * d_ff) + (d_ff * d_model) + (2 * d_model)
    encoder_layer_params = mha_params + ffn_params
    encoder_layers_params_x = encoder_layer_params * 3
    
    params += encoder_layers_params_x
    print(f"3个EncoderLayer: 参数={encoder_layers_params_x:,}")
    
    # FLOPs计算
    mha_flops = (d_k * datanum * datanum) + (datanum * datanum) + (datanum * datanum * d_v) + \
                (datanum * (d_model * d_k * 3 + d_v * d_model))
    ffn_flops = datanum * d_model * d_ff * 2
    encoder_layer_flops = mha_flops + ffn_flops
    encoder_layers_flops_x = encoder_layer_flops * 3
    flops += encoder_layers_flops_x
    print(f"3个EncoderLayer FLOPs: {encoder_layers_flops_x:,}")
    
    # 2. Stream_cond 分支 (相同)
    print("\n--- Stream_cond 分支 ---")
    params += conv_params_x + encoder_layers_params_x
    flops += conv_flops_x + encoder_layers_flops_x
    print(f"Conv1D层: 参数={conv_params_x:,}, FLOPs={conv_flops_x:,}")
    print(f"3个EncoderLayer: 参数={encoder_layers_params_x:,}, FLOPs={encoder_layers_flops_x:,}")
    
    # 3. PositionalEncoding
    print("\n--- PositionalEncoding ---")
    pos_enc_flops = feats * 2
    flops += pos_enc_flops
    print(f"PositionalEncoding FLOPs: {pos_enc_flops}")
    
    # 4. Bridge (FiLM layers x4, 调整为d_model=400)
    print("\n--- Bridge (FiLM layers) ---")
    film_params_per_layer = (1 * d_model) * 2
    film_params_total = film_params_per_layer * 4
    
    film_flops_per_layer = 2 * d_model * datanum
    film_flops_total = film_flops_per_layer * 4
    
    params += film_params_total
    flops += film_flops_total
    print(f"4个FiLM层: 参数={film_params_total:,}, FLOPs={film_flops_total:,}")
    
    # 5. 输出卷积层
    print("\n--- 输出卷积层 ---")
    conv_out1_params = feats * feats * 3 + feats
    conv_out2_params = feats * 1 * 3 + 1
    conv_out_params = conv_out1_params + conv_out2_params
    
    conv_out1_flops = feats * feats * 3 * datanum
    conv_out2_flops = feats * 1 * 3 * datanum
    conv_out_flops = conv_out1_flops + conv_out2_flops
    
    params += conv_out_params
    flops += conv_out_flops
    print(f"输出Conv层: 参数={conv_out_params:,}, FLOPs={conv_out_flops:,}")
    
    print(f"\n总参数量: {params:,}")
    print(f"总FLOPs: {flops:,}")
    print(f"总FLOPs (GFLOPs): {flops/1e9:.3f}")
    
    return params, flops

def analyze_ddpm_wrapper_overhead():
    """
    分析DDPM包装器的额外开销
    """
    print("\n" + "="*70)
    print("DDPM包装器额外开销分析")
    print("="*70)
    
    print("DDPM包装器主要包含:")
    print("1. 噪声调度参数 (betas, alphas等) - 存储开销，无计算参数")
    print("2. 扩散过程的前向和反向计算")
    print("3. 条件生成的采样循环")
    
    print("\n主要计算开销:")
    print("- 前向扩散: q_sample() - 线性组合操作")
    print("- 反向去噪: p_sample_loop() - 多步迭代调用base_model")
    print("- 每步调用base_model一次，总体FLOPs = base_model_FLOPs × 去噪步数")
    
    typical_steps = 50  # 典型的去噪步数
    print(f"\n假设去噪步数: {typical_steps}")
    print("总计算量 ≈ DualBranchDenoisingModel的FLOPs × 50")

def main():
    """主函数：分析EEGDfus模型系列"""
    datanum = 512  # 典型的EEG数据长度
    feats = 64     # 默认特征维度
    
    print("EEGDfus (扩散模型) 参数量和FLOPs分析")
    print(f"假设输入数据长度: {datanum}")
    print(f"特征维度: {feats}")
    
    # 分析两个模型变体
    eegdnet_params, eegdnet_flops = analyze_DualBranchDenoisingModel_EEGDNet(datanum, feats)
    ssed_params, ssed_flops = analyze_DualBranchDenoisingModel_SSED(datanum, feats)
    
    # 分析DDPM开销
    analyze_ddpm_wrapper_overhead()
    
    # 汇总比较
    print("\n" + "="*80)
    print("EEGDfus模型对比汇总")
    print("="*80)
    
    models = [
        ("DualBranch_EEGDNet", eegdnet_params, eegdnet_flops),
        ("DualBranch_SSED", ssed_params, ssed_flops),
    ]
    
    print(f"{'模型版本':<25} {'参数量':<15} {'单步FLOPs':<15} {'单步GFLOPs':<12}")
    print("-" * 75)
    
    for name, params, flops in models:
        print(f"{name:<25} {params:<15,} {flops:<15,} {flops/1e9:<12.3f}")
    
    print(f"\n{'模型版本':<25} {'50步总FLOPs':<15} {'50步总GFLOPs':<15}")
    print("-" * 60)
    for name, params, flops in models:
        total_flops = flops * 50
        print(f"{name:<25} {total_flops:<15,} {total_flops/1e9:<15.1f}")
    
    # 技术特点分析
    print("\n" + "="*80)
    print("EEGDfus技术特点分析")
    print("="*80)
    
    print("1. 双分支架构:")
    print("   - stream_x: 处理噪声输入")
    print("   - stream_cond: 处理条件信息(噪声EEG)")
    print("   - FiLM层实现跨模态特征调节")
    
    print("\n2. 核心技术组件:")
    print("   - Self-Attention (Transformer): 捕获长程依赖")  
    print("   - 扩散模型(DDPM): 渐进式去噪")
    print("   - 位置编码: 噪声级别嵌入")
    print("   - 条件生成: 基于噪声EEG的引导去噪")
    
    print("\n3. 计算复杂度:")
    print("   - EEGDNet版本: 更高维度(d_model=512)，精度可能更高")
    print("   - SSED版本: 较低维度(d_model=400)，计算效率更高")
    print("   - 扩散过程: 需要多步迭代，总计算量大")
    
    print("\n4. 适用场景:")
    print("   - 高质量EEG信号恢复")
    print("   - 对计算资源要求较高")
    print("   - 适合离线处理或高性能设备")

if __name__ == "__main__":
    main()