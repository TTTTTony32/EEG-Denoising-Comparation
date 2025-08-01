import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体和样式
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})

def read_model_data(file_path):
    """
    从CSV文件读取模型性能数据
    
    Args:
        file_path (str): CSV文件路径
    
    Returns:
        pandas.DataFrame: 包含模型数据的DataFrame
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"成功读取数据文件: {file_path}")
        print(f"数据形状: {df.shape}")
        print("\n数据预览:")
        print(df.head())
        return df
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def plot_rmse_flops_scatter(df, save_path=None):
    """
    绘制RMSE-FLOPs散点图
    
    Args:
        df (pandas.DataFrame): 模型数据
        save_path (str): 保存图片的路径
    """
    plt.figure(figsize=(8, 7))
    
    # 创建散点图
    scatter = plt.scatter(df['FLOPs'], df['RMSE'], 
                         s=100, alpha=0.7, 
                         c=range(len(df)), 
                         cmap='viridis', 
                         edgecolors='black', 
                         linewidth=1)
    
    # 添加模型名称标签
    for i, model in enumerate(df['模型名称']):
        # 单独调整EEGDfus标签位置到左边
        if model == 'EEGDfus':
            xytext_offset = (-7, -5)
            ha_align = 'right'
        else:
            xytext_offset = (7, -5)
            ha_align = 'left'
            
        plt.annotate(model, 
                    (df['FLOPs'].iloc[i], df['RMSE'].iloc[i]),
                    xytext=xytext_offset, 
                    textcoords='offset points',
                    fontsize=18,
                    ha=ha_align)

    plt.xlabel('FLOPs', fontsize=18, fontweight='bold')
    plt.ylabel('RMSE', fontsize=18, fontweight='bold')
    plt.title('Model Performance: RMSE vs FLOPs', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 设置科学计数法显示FLOPs
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # 添加对角线（从左上到右下）
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    # 计算对角线的起点和终点
    diagonal_x = [x_min, x_max]
    diagonal_y = [y_max, y_min]  # 左上到右下
    plt.plot(diagonal_x, diagonal_y, '--', color='gray', alpha=0.4, linewidth=1, label='Reference Line')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RMSE-FLOPs散点图已保存到: {save_path}")
    
    plt.show()

def plot_rmse_params_scatter(df, save_path=None):
    """
    绘制RMSE-参数量散点图
    
    Args:
        df (pandas.DataFrame): 模型数据
        save_path (str): 保存图片的路径
    """
    plt.figure(figsize=(8, 7))
    
    # 创建散点图
    scatter = plt.scatter(df['参数量'], df['RMSE'], 
                         s=100, alpha=0.7, 
                         c=range(len(df)), 
                         cmap='plasma', 
                         edgecolors='black', 
                         linewidth=1)
    
    # 添加模型名称标签
    for i, model in enumerate(df['模型名称']):
        # 单独调整EEGDiR标签位置到左边
        if model == 'EEGDiR':
            xytext_offset = (-7, -5)
            ha_align = 'right'
        else:
            xytext_offset = (7, -5)
            ha_align = 'left'
            
        plt.annotate(model, 
                    (df['参数量'].iloc[i], df['RMSE'].iloc[i]),
                    xytext=xytext_offset, 
                    textcoords='offset points',
                    fontsize=18,
                    ha=ha_align)
    
    plt.xlabel('Parameters Count', fontsize=18, fontweight='bold')
    plt.ylabel('RMSE', fontsize=18, fontweight='bold')
    plt.title('Model Performance: RMSE vs Parameters Count', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 设置科学计数法显示参数量
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # 添加对角线（从左上到右下）
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    # 计算对角线的起点和终点
    diagonal_x = [x_min, x_max]
    diagonal_y = [y_max, y_min]  # 左上到右下
    plt.plot(diagonal_x, diagonal_y, '--', color='gray', alpha=0.4, linewidth=1, label='Reference Line')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RMSE-参数量散点图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    # 文件路径
    data_file = 'model_performance.csv'
    
    # 读取数据
    df = read_model_data(data_file)
    
    if df is not None:
        # 绘制单独的散点图
        plot_rmse_flops_scatter(df, 'results/rmse_flops_scatter.pdf')
        plot_rmse_params_scatter(df, 'results/rmse_params_scatter.pdf')
        
        print("\n所有图表已生成完成！")
    else:
        print("无法读取数据文件，请检查文件路径和格式。")

if __name__ == "__main__":
    main()
