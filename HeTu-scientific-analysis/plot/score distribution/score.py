import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def generate_fraction_plot(folder_path, min_score=0.5):
    """
    使用 Matplotlib 绘制符合科研标准的 Fraction 分布图
    """
    # 1. 获取并过滤文件路径
    all_paths = glob.glob(os.path.join(folder_path, "*.csv"))
    
    exclude_files = {
        'processed_wcs_20147.csv', 'processed_wcs_20161.csv', 'processed_wcs_20171.csv', 'processed_wcs_20172.csv', 'processed_wcs_20175.csv', 'processed_wcs_20776.csv',
        'processed_wcs_22654.csv', 'processed_wcs_22661.csv', 'processed_wcs_22672.csv', 'processed_wcs_22882.csv',
        'processed_wcs_22900.csv', 'processed_wcs_25453.csv', 'processed_wcs_25468.csv', 'processed_wcs_25479.csv',
        'processed_wcs_25498.csv', 'processed_wcs_25703.csv', 'processed_wcs_20261.csv', 'processed_wcs_20263.csv',
        'processed_wcs_20265.csv', 'processed_wcs_20287.csv', 'processed_wcs_20479.csv', 'processed_wcs_20488.csv',
        'processed_wcs_20640.csv', 'processed_wcs_20306.csv', 'processed_wcs_20534.csv', 'processed_wcs_20615.csv',
        'processed_wcs_33090.csv'
    }
    
    if not all_paths:
        print(f"Error: No CSV files found in '{folder_path}'.")
        return

    # 2. 批量读取数据
    data_list = []
    for file_path in all_paths:
        file_name = os.path.basename(file_path)
        if file_name in exclude_files:
            continue
        try:
            # 只加载必要的列
            df = pd.read_csv(file_path, usecols=['label', 'score'])
            data_list.append(df)
        except Exception as e:
            print(f"Skipping {file_name}: {e}")

    if not data_list:
        print("Error: No data to plot after filtering.")
        return

    full_df = pd.concat(data_list, ignore_index=True)
    # 过滤 score 区间
    filtered_df = full_df[(full_df['score'] >= min_score) & (full_df['score'] <= 1.0)]

    # 3. 设置科研绘图风格
    plt.rcParams.update({
        'font.size': 12,
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # 标签映射
    label_map = {0: 'CJ', 1: 'CS', 2: 'FRI', 3: 'FRII'}
    
    # 获取唯一的标签并排序
    unique_labels = sorted(filtered_df['label'].unique())
    
    # 使用专业的色彩循环
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 4. 绘图逻辑：阶梯状直方图 (无填充)
    # 增加 bin 数量到 50，让图像更平滑
    num_bins = 30
    bins = np.linspace(min_score, 1.0, num_bins + 1)

    for idx, label_id in enumerate(unique_labels):
        subset = filtered_df[filtered_df['label'] == label_id]['score']
        if subset.empty:
            continue
        
        # 获取映射后的名称
        display_name = label_map.get(label_id, f'Label {label_id}')

        # 绘制阶梯直方图边缘线
        # density=False, weights=np.ones/len(subset) 用于手动计算 Fraction
        plt.hist(subset, bins=bins, 
                 weights=np.ones(len(subset)) / len(subset),
                 histtype='step', 
                 linewidth=2, 
                 label=display_name, 
                 color=colors[idx % len(colors)])

    # 5. 图表精修 (移除标题和网格)
    plt.xlabel('Score', fontsize=12, labelpad=10)
    plt.ylabel('Fraction', fontsize=12, labelpad=10)
    plt.xlim(min_score, 1.0)
    plt.ylim(0, None)

    # 图例放在左上角（通常不会遮挡右侧的高分数据），移除边框
    plt.legend(loc='upper left', frameon=False, fontsize=12)

    # 移除顶部和右侧的刻度线（可选，部分期刊要求）
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # 6. 保存高质量图片
    output_path = 'scientific_score_fraction.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Scientific plot saved to: {output_path}")

if __name__ == "__main__":
    # 执行当前目录下的分析
    generate_fraction_plot("/groups/hetu_ai/home/share/HeTu/pjlab/HeTu-FM-train/cateloge_creation/bbox_overlap_removal_maskb", min_score=0.5)