import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv_files(path):
    scores = []
    # 需排除的CSV文件名列表（注意带.csv后缀）
    excluded_files = [
        '20161.csv', '20171.csv', '20172.csv', '20175.csv', '20776.csv',
        '22654.csv', '22661.csv', '22672.csv', '22882.csv',
        '22900.csv', '25453.csv', '25468.csv', '25479.csv',
        '25498.csv', '25703.csv', '20261.csv', '20263.csv',
        '20265.csv', '20287.csv', '20479.csv', '20488.csv',
        '20640.csv', '20306.csv', '20534.csv', '20615.csv',
        '33090.csv'
    ]
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv') and file not in excluded_files:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                scores.extend(df['score'].tolist())
    return scores


def filter_scores(scores):
    return [score for score in scores if score >= 0.3]


def plot_histogram_and_midpoints(scores):
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    n, bins, patches = plt.hist(scores, bins=50, density=True, alpha=0.6, color='g', label='Histogram')
    
    # 计算直方图中点并连线
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, n, 'b-o', linewidth=2, markersize=6, label='Midpoints Connection')
    
    # 计算并标记平均得分
    mean_score = np.mean(scores)
    plt.axvline(x=mean_score, color='k', linestyle='--', linewidth=2, 
                label=f'Mean Score: {mean_score:.4f}')
    
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Score Distribution and Midpoints Connection', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # 保存图像的路径（沿用之前设定的目录）
    save_dir = 'home/ydai240628/analysis_hetu/pic'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'score_distribution.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 替换为您提供的路径
    path = '/home/ydai240628/analysis_hetu/code/only_label/output_internimage_0722/'
    all_scores = read_csv_files(path)
    filtered_scores = filter_scores(all_scores)
    
    if not filtered_scores:
        print("No valid scores after filtering.")
    else:
        try:
            plot_histogram_and_midpoints(filtered_scores)
            print(f"Average Score: {np.mean(filtered_scores):.4f}")
        except Exception as e:
            print(f"An error occurred: {e}")