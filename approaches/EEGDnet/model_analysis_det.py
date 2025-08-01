"""
EEGDnet (DeT - Denoising Transformer) 模型参数量和FLOPs分析
基于标准Transformer架构的EEG去噪模型
"""

def analyze_FeedForward(dim, seq_len):
    """
    分析FeedForward模块
    """
    hidden_dim = 2 * dim  # 默认是2倍扩展
    
    # 参数量
    # Linear(dim, hidden_dim) + Linear(hidden_dim, dim)
    linear1_params = dim * hidden_dim + hidden_dim  # weights + bias
    linear2_params = hidden_dim * dim + dim
    total_params = linear1_params + linear2_params
    
    # FLOPs
    # Linear1: seq_len × dim × hidden_dim
    # Linear2: seq_len × hidden_dim × dim 
    # PReLU: seq_len × hidden_dim (近似)
    linear1_flops = seq_len * dim * hidden_dim
    linear2_flops = seq_len * hidden_dim * dim
    prelu_flops = seq_len * hidden_dim
    total_flops = linear1_flops + linear2_flops + prelu_flops
    
    return total_params, total_flops

def analyze_Attention(dim, heads, seq_len):
    """
    分析Attention模块 (Multi-Head Self-Attention)
    """
    dim_head = dim  # 在这个实现中，dim_head = dim
    inner_dim = dim_head * heads
    
    # 参数量
    # to_qkv: Linear(dim, inner_dim * 3, bias=False)
    qkv_params = dim * (inner_dim * 3)
    
    # to_out: Linear(inner_dim, dim) (当heads > 1时)
    if heads > 1:
        out_params = inner_dim * dim + dim  # weights + bias
    else:
        out_params = 0  # Identity layer
    
    total_params = qkv_params + out_params
    
    # FLOPs
    # QKV计算: seq_len × dim × (inner_dim * 3)
    qkv_flops = seq_len * dim * (inner_dim * 3)
    
    # Attention计算: 
    # Q @ K^T: heads × seq_len × dim_head × seq_len  
    qk_flops = heads * seq_len * dim_head * seq_len
    
    # Softmax: heads × seq_len × seq_len (近似)
    softmax_flops = heads * seq_len * seq_len
    
    # Attention @ V: heads × seq_len × seq_len × dim_head
    av_flops = heads * seq_len * seq_len * dim_head
    
    # Output projection: seq_len × inner_dim × dim (如果需要)
    if heads > 1:
        out_flops = seq_len * inner_dim * dim
    else:
        out_flops = 0
    
    total_flops = qkv_flops + qk_flops + softmax_flops + av_flops + out_flops
    
    return total_params, total_flops

def analyze_Transformer(dim, depth, heads, seq_len):
    """
    分析Transformer模块
    """
    total_params = 0
    total_flops = 0
    
    # LayerNorm参数 (每层有2个LayerNorm)
    layer_norm_params = depth * 2 * (2 * dim)  # scale + bias
    total_params += layer_norm_params
    
    # LayerNorm FLOPs (每层2个LayerNorm)
    layer_norm_flops = depth * 2 * (4 * dim * seq_len)  # 近似计算
    total_flops += layer_norm_flops
    
    # 每层的Attention和FeedForward
    for layer in range(depth):
        # Attention
        attn_params, attn_flops = analyze_Attention(dim, heads, seq_len)
        total_params += attn_params
        total_flops += attn_flops
        
        # FeedForward
        ff_params, ff_flops = analyze_FeedForward(dim, seq_len)
        total_params += ff_params
        total_flops += ff_flops
    
    return total_params, total_flops

def analyze_DeT(seq_len=512, patch_len=64, depth=6, heads=1, dropout=0.1):
    """
    分析完整的DeT (Denoising Transformer) 模型
    """
    print("="*70)
    print("EEGDNET - DeT (Denoising Transformer) 模型分析")
    print("="*70)
    print(f"输入维度: (batch_size, {seq_len})")
    print(f"模型参数: patch_len={patch_len}, depth={depth}, heads={heads}")
    
    # 检查输入合法性
    if seq_len % patch_len != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})")
    
    num_patches = seq_len // patch_len
    patch_dim = patch_len
    dim = patch_dim  # 在DeT中，dim = patch_dim
    
    print(f"分块参数: num_patches={num_patches}, patch_dim={patch_dim}, dim={dim}")
    
    total_params = 0
    total_flops = 0
    
    # 1. Patch Embedding
    print("\n--- Patch Embedding ---")
    # Rearrange操作不需要参数
    # Linear(patch_dim, dim)
    patch_embed_params = patch_dim * dim + dim
    patch_embed_flops = num_patches * patch_dim * dim  # 对每个patch进行线性变换
    
    total_params += patch_embed_params
    total_flops += patch_embed_flops
    print(f"Patch Embedding: 参数={patch_embed_params:,}, FLOPs={patch_embed_flops:,}")
    
    # 2. 位置编码
    print("\n--- 位置编码 ---")
    pos_embed_params = num_patches * dim  # nn.Parameter
    total_params += pos_embed_params
    # 位置编码加法操作的FLOPs
    pos_embed_flops = num_patches * dim
    total_flops += pos_embed_flops
    print(f"位置编码: 参数={pos_embed_params:,}, FLOPs={pos_embed_flops:,}")
    
    # 3. Transformer
    print("\n--- Transformer ---")
    transformer_params, transformer_flops = analyze_Transformer(
        dim, depth, heads, num_patches
    )
    total_params += transformer_params
    total_flops += transformer_flops
    print(f"Transformer ({depth}层): 参数={transformer_params:,}, FLOPs={transformer_flops:,}")
    
    # 4. 序列重构
    print("\n--- 序列重构 ---")
    # Linear(dim, patch_dim)
    seq_recon_params = dim * patch_dim + patch_dim
    seq_recon_flops = num_patches * dim * patch_dim  # 对每个patch进行线性变换
    
    total_params += seq_recon_params
    total_flops += seq_recon_flops
    print(f"序列重构: 参数={seq_recon_params:,}, FLOPs={seq_recon_flops:,}")
    
    print(f"\n总参数量: {total_params:,}")
    print(f"总FLOPs: {total_flops:,}")
    print(f"总FLOPs (MFLOPs): {total_flops/1e6:.3f}")
    print(f"总FLOPs (GFLOPs): {total_flops/1e9:.3f}")
    
    return total_params, total_flops

def analyze_attention_complexity(seq_len, patch_len, heads):
    """
    分析注意力机制的复杂度特点
    """
    print("\n" + "="*70)
    print("注意力机制复杂度分析")
    print("="*70)
    
    num_patches = seq_len // patch_len
    
    # 原始序列的Self-Attention复杂度: O(L^2)
    original_complexity = seq_len ** 2
    
    # Patch-based Attention复杂度: O(P^2) where P = L/patch_len  
    patch_complexity = num_patches ** 2
    
    # 复杂度减少倍数
    reduction = original_complexity / patch_complexity
    
    print(f"序列长度: {seq_len}")
    print(f"分块大小: {patch_len}")
    print(f"分块数量: {num_patches}")
    print(f"\n复杂度对比:")
    print(f"原始Self-Attention: O(L²) = O({seq_len}²) = {original_complexity:,}")
    print(f"Patch-based Attention: O(P²) = O({num_patches}²) = {patch_complexity:,}")
    print(f"复杂度减少: {reduction:.1f}x")
    
    print(f"\nDeT设计特点:")
    print(f"1. 分块策略: 将长序列分成{num_patches}个patch，每个patch长度{patch_len}")
    print(f"2. 注意力计算: 在patch级别计算，而不是样本级别")
    print(f"3. 位置编码: 保持patch间的位置关系")
    print(f"4. 计算效率: 大幅降低二次复杂度的影响")

def compare_with_other_models():
    """
    与其他模型架构进行对比
    """
    print("\n" + "="*80)
    print("与其他架构的对比分析")
    print("="*80)
    
    # 分析不同配置的DeT
    configs = [
        {"name": "DeT-Small", "seq_len": 512, "patch_len": 64, "depth": 4, "heads": 1},
        {"name": "DeT-Base", "seq_len": 512, "patch_len": 64, "depth": 6, "heads": 1}, 
        {"name": "DeT-Large", "seq_len": 512, "patch_len": 32, "depth": 8, "heads": 2},
        {"name": "DeT-Deep", "seq_len": 512, "patch_len": 64, "depth": 12, "heads": 1},
    ]
    
    print(f"{'模型配置':<15} {'参数量':<15} {'FLOPs(M)':<12} {'Patches':<8} {'深度':<6}")
    print("-" * 70)
    
    for config in configs:
        try:
            params, flops = analyze_DeT(
                seq_len=config["seq_len"],
                patch_len=config["patch_len"], 
                depth=config["depth"],
                heads=config["heads"]
            )
            num_patches = config["seq_len"] // config["patch_len"]
            print(f"{config['name']:<15} {params:<15,} {flops/1e6:<12.1f} {num_patches:<8} {config['depth']:<6}")
        except Exception as e:
            print(f"{config['name']:<15} Error: {str(e)}")

def main():
    """主函数：分析EEGDnet模型"""
    # 默认配置 (基于Network_structure.py的实现)
    seq_len = 512       # EEG序列长度
    patch_len = 64      # 分块大小
    depth = 6           # Transformer层数
    heads = 1           # 注意力头数 (单头注意力)
    dropout = 0.1       # Dropout比率
    
    print("EEGDnet (基于Transformer的EEG去噪) 模型分析")
    print(f"默认配置参数:")
    print(f"- 序列长度: {seq_len}")
    print(f"- 分块大小: {patch_len}")  
    print(f"- Transformer层数: {depth}")
    print(f"- 注意力头数: {heads}")
    print(f"- Dropout比率: {dropout}")
    
    # 分析默认配置
    det_params, det_flops = analyze_DeT(seq_len, patch_len, depth, heads, dropout)
    
    # 分析注意力复杂度
    analyze_attention_complexity(seq_len, patch_len, heads)
    
    # 模型配置对比
    compare_with_other_models()
    
    # 技术特点总结
    print("\n" + "="*80)
    print("EEGDnet (DeT) 技术特点总结")
    print("="*80)
    
    print("1. 架构设计:")
    print("   - 基于标准Transformer Encoder")
    print("   - Patch-based输入处理")
    print("   - 位置编码保持序列信息")
    print("   - 端到端的序列重构")
    
    print("\n2. 核心技术:")
    print("   - Self-Attention: 捕获长程依赖关系")
    print("   - FeedForward: 非线性特征变换")
    print("   - LayerNorm + 残差连接: 稳定训练")
    print("   - Patch切分: 降低计算复杂度")
    
    print("\n3. 计算特点:")
    print("   - 注意力复杂度: O(P²) where P = seq_len/patch_len")
    print("   - 参数共享: 所有position使用相同的变换")
    print("   - 并行计算: 支持高效的并行处理")
    
    print("\n4. 适用场景:")
    print("   - 中等长度EEG序列处理")
    print("   - 需要捕获长程依赖的任务")
    print("   - 对计算效率有一定要求的场景")
    print("   - 序列到序列的去噪任务")

if __name__ == "__main__":
    main()