import os
import pandas as pd

def process_csv_files(folder_path, output_csv_path):
    all_results = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    print(f"{file_path}: {len(df)}")
                    
                    # 检查必要的列是否存在
                    if 'label' not in df.columns:
                        print(f"{file_path} missing 'label' column")
                        continue
                    if 'score' not in df.columns:
                        print(f"{file_path} missing 'score' column")
                        continue
                    
                    # 筛选score >= 0.5的数据
                    filtered_df = df[df['score'] >= 0.5]
                    
                    # 统计各label数量（原始数据）
                    label_0_count = df[df['label'] == 0]['label'].count()
                    label_1_count = df[df['label'] == 1]['label'].count()
                    label_2_count = df[df['label'] == 2]['label'].count()
                    label_3_count = df[df['label'] == 3]['label'].count()
                    
                    # 统计score>=0.5的各label数量
                    filtered_label_0_count = filtered_df[filtered_df['label'] == 0]['label'].count()
                    filtered_label_1_count = filtered_df[filtered_df['label'] == 1]['label'].count()
                    filtered_label_2_count = filtered_df[filtered_df['label'] == 2]['label'].count()
                    filtered_label_3_count = filtered_df[filtered_df['label'] == 3]['label'].count()
                    
                    SBID = file
                    print(f"SBID:{SBID}, "
                          f"label_0: {label_0_count} ({filtered_label_0_count} >=0.5), "
                          f"label_1: {label_1_count} ({filtered_label_1_count} >=0.5), "
                          f"label_2: {label_2_count} ({filtered_label_2_count} >=0.5), "
                          f"label_3: {label_3_count} ({filtered_label_3_count} >=0.5)")

                    result = {
                        'SBID': SBID,
                        'label_0_count': label_0_count,
                        'label_1_count': label_1_count,
                        'label_2_count': label_2_count,
                        'label_3_count': label_3_count,
                        'label_0_count_filtered': filtered_label_0_count,
                        'label_1_count_filtered': filtered_label_1_count,
                        'label_2_count_filtered': filtered_label_2_count,
                        'label_3_count_filtered': filtered_label_3_count
                    }
                    all_results.append(result)

                except Exception as e:
                    print(f"{file_path} error: {e}")

    if all_results:
        result_df = pd.DataFrame(all_results)
        print(f"Results DataFrame shape: {result_df.shape}")
        result_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")
    else:
        print("No valid CSV files found or all files had errors.")

# 示例使用
folder_path = '/home/ydai240628/analysis_hetu/file/bbox_overlap_removal/output_internimage_0722/'
output_csv_path = '/home/ydai240628/analysis_hetu/file/count_internimage_0.5.csv'
process_csv_files(folder_path, output_csv_path)