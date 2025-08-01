"""
EEGDiR (RetNet) 模型参数量和FLOPs分析
基于retention机制的EEG去噪模型
"""

def analyze_SimpleRetention(hidden_size, head_size, v_dim, sequence_length):
    """
    分析单个SimpleRetention模块
    """
    params = 0
    flops = 0
    
    # 参数量计算
    # W_Q: hidden_size × head_size
    # W_K: hidden_size × head_size  
    # W_V: hidden_size × v_dim
    wq_params = hidden_size * head_size
    wk_params = hidden_size * head_size
    wv_params = hidden_size * v_dim
    params = wq_params + wk_params + wv_params
    
    # FLOPs计算
    # Q = X @ W_Q: batch_size × sequence_length × hidden_size @ hidden_size × head_size
    q_flops = sequence_length * hidden_size * head_size
    
    # K = X @ W_K: 同上
    k_flops = sequence_length * hidden_size * head_size
    
    # V = X @ W_V: batch_size × sequence_length × hidden_size @ hidden_size × v_dim
    v_flops = sequence_length * hidden_size * v_dim
    
    # Q @ K^T: batch_size × sequence_length × head_size @ head_size × sequence_length
    qk_flops = sequence_length * head_size * sequence_length
    
    # (Q @ K^T * D) @ V: batch_size × sequence_length × sequence_length @ sequence_length × v_dim
    output_flops = sequence_length * sequence_length * v_dim
    
    # XPOS旋转位置编码的额外计算
    xpos_flops = sequence_length * head_size * 4  # sin, cos, 旋转操作
    
    flops = q_flops + k_flops + v_flops + qk_flops + output_flops + xpos_flops
    
    return params, flops

def analyze_MultiScaleRetention(hidden_size, heads, double_v_dim, sequence_length):
    """
    分析MultiScaleRetention模块
    """
    v_dim = hidden_size * 2 if double_v_dim else hidden_size
    head_size = hidden_size // heads
    head_v_dim = hidden_size * 2 if double_v_dim else hidden_size
    
    params = 0
    flops = 0
    
    # 每个头的SimpleRetention
    retention_params = 0
    retention_flops = 0
    for i in range(heads):
        head_params, head_flops = analyze_SimpleRetention(hidden_size, head_size, head_v_dim, sequence_length)
        retention_params += head_params
        retention_flops += head_flops
    
    # W_G: hidden_size × v_dim
    wg_params = hidden_size * v_dim
    
    # W_O: v_dim × hidden_size
    wo_params = v_dim * hidden_size
    
    # GroupNorm参数: heads个组，每组v_dim/heads个参数，共2*v_dim个参数(scale+bias)
    group_norm_params = 2 * v_dim
    
    params = retention_params + wg_params + wo_params + group_norm_params
    
    # FLOPs计算
    # X @ W_G: sequence_length × hidden_size × v_dim
    wg_flops = sequence_length * hidden_size * v_dim
    
    # Swish激活函数: v_dim × sequence_length 
    swish_flops = sequence_length * v_dim * 2  # sigmoid + multiply
    
    # (swish(X @ W_G) * Y) @ W_O: sequence_length × v_dim × hidden_size
    output_flops = sequence_length * v_dim * hidden_size
    
    # GroupNorm: 大约 4 × v_dim × sequence_length 的操作
    group_norm_flops = 4 * v_dim * sequence_length
    
    flops = retention_flops + wg_flops + swish_flops + output_flops + group_norm_flops
    
    return params, flops

def analyze_RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim, sequence_length):
    """
    分析RetNet网络
    """
    params = 0
    flops = 0
    
    for layer in range(layers):
        # MultiScaleRetention
        retention_params, retention_flops = analyze_MultiScaleRetention(
            hidden_dim, heads, double_v_dim, sequence_length
        )
        params += retention_params
        flops += retention_flops
        
        # FFN: Linear(hidden_dim, ffn_size) + GELU + Dropout + Linear(ffn_size, hidden_dim)
        ffn_params = (hidden_dim * ffn_size) + ffn_size + (ffn_size * hidden_dim) + hidden_dim
        ffn_flops = sequence_length * ((hidden_dim * ffn_size) + (ffn_size * hidden_dim))
        gelu_flops = sequence_length * ffn_size * 8  # GELU近似计算复杂度
        
        params += ffn_params
        flops += ffn_flops + gelu_flops
        
        # LayerNorm x2: 每个LayerNorm有2*hidden_dim个参数
        layer_norm_params = 2 * (2 * hidden_dim)  # 两个LayerNorm
        layer_norm_flops = 2 * (4 * hidden_dim * sequence_length)  # 两个LayerNorm的计算
        
        params += layer_norm_params
        flops += layer_norm_flops
    
    return params, flops

def analyze_DiR(layers, hidden_dim, ffn_size, heads, double_v_dim, seq_len, mini_seq):
    """
    分析完整的DiR模型
    """
    print("="*70)
    print("EEGDIR - DiR (Retention-based Denoiser) 模型分析")
    print("="*70)
    print(f"输入维度: (batch_size, {seq_len})")
    print(f"模型参数: layers={layers}, hidden_dim={hidden_dim}, ffn_size={ffn_size}")
    print(f"multi-head参数: heads={heads}, double_v_dim={double_v_dim}")
    print(f"分块参数: mini_seq={mini_seq}")
    
    if seq_len % mini_seq != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by mini_seq ({mini_seq})")
    
    params = 0
    flops = 0
    
    # 1. Patchify网络
    print("\n--- Patchify 网络 ---")
    # Linear(seq_len, mini_seq * hidden_dim)
    patch_linear1_params = seq_len * (mini_seq * hidden_dim) + (mini_seq * hidden_dim)
    patch_linear1_flops = seq_len * (mini_seq * hidden_dim)
    
    # Linear(mini_seq * hidden_dim, mini_seq * hidden_dim) 
    patch_linear2_params = (mini_seq * hidden_dim) * (mini_seq * hidden_dim) + (mini_seq * hidden_dim)
    patch_linear2_flops = (mini_seq * hidden_dim) * (mini_seq * hidden_dim)
    
    # GELU激活函数 (两次)
    gelu_flops = (mini_seq * hidden_dim) * 8 * 2  # GELU近似复杂度
    
    patch_params = patch_linear1_params + patch_linear2_params
    patch_flops = patch_linear1_flops + patch_linear2_flops + gelu_flops
    
    params += patch_params
    flops += patch_flops
    print(f"Patchify: 参数={patch_params:,}, FLOPs={patch_flops:,}")
    
    # 2. RetNet网络
    print("\n--- RetNet 网络 ---")
    # 序列长度变为mini_seq (因为经过了chunk操作)
    retnet_params, retnet_flops = analyze_RetNet(
        layers, hidden_dim, ffn_size, heads, double_v_dim, mini_seq
    )
    
    params += retnet_params
    flops += retnet_flops
    print(f"RetNet ({layers}层): 参数={retnet_params:,}, FLOPs={retnet_flops:,}")
    
    # 3. 输出网络
    print("\n--- 输出网络 ---")
    # Linear(hidden_dim, seq_len // mini_seq)
    out_params = hidden_dim * (seq_len // mini_seq) + (seq_len // mini_seq)
    out_flops = mini_seq * hidden_dim * (seq_len // mini_seq)  # mini_seq个序列位置
    
    params += out_params
    flops += out_flops
    print(f"输出网络: 参数={out_params:,}, FLOPs={out_flops:,}")
    
    print(f"\n总参数量: {params:,}")
    print(f"总FLOPs: {flops:,}")
    print(f"总FLOPs (MFLOPs): {flops/1e6:.3f}")
    print(f"总FLOPs (GFLOPs): {flops/1e9:.3f}")
    
    return params, flops

def analyze_retention_complexity(seq_len, mini_seq, hidden_dim, heads):
    """
    分析retention机制的计算复杂度特点
    """
    print("\n" + "="*70)
    print("Retention机制复杂度分析")
    print("="*70)
    
    # 传统Attention复杂度: O(L^2 * d)
    attention_complexity = (seq_len ** 2) * hidden_dim
    
    # Retention复杂度: O(L * d^2) (并行模式)
    retention_complexity = seq_len * (hidden_dim ** 2)
    
    # DiR中的实际复杂度 (使用mini_seq)
    dir_complexity = mini_seq * (hidden_dim ** 2)
    
    print(f"序列长度: {seq_len}, 隐藏维度: {hidden_dim}")
    print(f"分块大小: {mini_seq}")
    print(f"\n复杂度对比:")
    print(f"传统Attention: O(L²×d) ≈ {attention_complexity:,}")
    print(f"Retention(完整): O(L×d²) ≈ {retention_complexity:,}")  
    print(f"DiR(分块): O(mini_seq×d²) ≈ {dir_complexity:,}")
    
    # 复杂度比较
    if dir_complexity < attention_complexity:
        reduction = attention_complexity / dir_complexity
        print(f"\nDiR相比传统Attention计算量减少: {reduction:.1f}x")
    
    print(f"\nRetention机制特点:")
    print(f"1. 线性复杂度: O(L×d²) vs Attention的O(L²×d)")
    print(f"2. 支持递归计算: 可以实现流式处理")
    print(f"3. 分块处理: 进一步降低内存占用")
    print(f"4. 多尺度设计: {heads}个不同衰减率的头")

def main():
    """主函数：分析EEGDiR模型"""
    # 模型配置 (基于retnet.py中的默认参数)
    layers = 8          # RetNet层数
    hidden_dim = 512    # 隐藏维度
    ffn_size = 1024     # FFN维度  
    heads = 8           # 多头数量
    double_v_dim = False # 是否双倍V维度
    seq_len = 512       # 输入序列长度
    mini_seq = 16       # 分块大小
    
    print("EEGDiR (基于RetNet的EEG去噪) 模型分析")
    print(f"配置参数总览:")
    print(f"- 网络层数: {layers}")
    print(f"- 隐藏维度: {hidden_dim}")
    print(f"- FFN维度: {ffn_size}")
    print(f"- 注意力头数: {heads}")
    print(f"- 序列长度: {seq_len}")
    print(f"- 分块大小: {mini_seq}")
    
    # 分析模型
    dir_params, dir_flops = analyze_DiR(
        layers, hidden_dim, ffn_size, heads, double_v_dim, seq_len, mini_seq
    )
    
    # 分析复杂度特点
    analyze_retention_complexity(seq_len, mini_seq, hidden_dim, heads)
    
    # 与其他架构对比
    print("\n" + "="*80)
    print("与其他架构的对比分析")
    print("="*80)
    
    # 估算等效Transformer的参数量作为对比
    # 标准Transformer: Multi-head Attention + FFN + LayerNorm
    transformer_params_per_layer = (
        # Multi-head Attention: Q,K,V,O矩阵
        4 * (hidden_dim * hidden_dim) +
        # FFN: 两个线性层
        (hidden_dim * ffn_size) + (ffn_size * hidden_dim) +
        # LayerNorm: 两个LayerNorm
        4 * hidden_dim
    )
    transformer_total = transformer_params_per_layer * layers
    
    print(f"模型对比:")
    print(f"{'架构':<20} {'参数量':<15} {'相对比例':<10}")
    print("-" * 50)
    print(f"{'DiR (RetNet)':<20} {dir_params:<15,} {'1.0x':<10}")
    print(f"{'等效Transformer':<20} {transformer_total:<15,} {transformer_total/dir_params:<10.1f}x")
    
    print(f"\nEEGDiR技术特点:")
    print(f"1. Retention机制: 线性复杂度，支持并行和递归计算")
    print(f"2. 分块处理: 降低内存占用，适合长序列")
    print(f"3. 多尺度建模: 不同衰减率捕获不同时间尺度特征")
    print(f"4. 位置编码: XPOS旋转位置编码，处理相对位置信息")
    print(f"5. 适用场景: 实时EEG处理，长序列建模")

if __name__ == "__main__":
    main()