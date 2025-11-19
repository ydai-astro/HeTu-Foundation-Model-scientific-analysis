# -*- coding: utf-8 -*-
import os
import pandas as pd


def count_low_scores(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                if 'score' in df.columns:
                    total_count = len(df['score'])
                    low_score_count = (df['score'] < 0.1).sum()
                    low_score_ratio = low_score_count / total_count if total_count > 0 else 0
                    results.append({
                        'filename': filename,
                        'low_score_count': low_score_count,
                        'low_score_ratio': low_score_ratio
                    })
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
    return results


def save_results_to_csv(results, output_path):
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    folder_path = '/home/ydai240628/analysis_hetu/code/only_label/output_internimage_0722/'
    output_path = '/home/ydai240628/analysis_hetu/file/output_lowscore.csv'
    results = count_low_scores(folder_path)
    save_results_to_csv(results, output_path)
    