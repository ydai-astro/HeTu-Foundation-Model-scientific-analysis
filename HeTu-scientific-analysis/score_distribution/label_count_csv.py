# -*- coding: utf-8 -*-
import os
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_labels_in_csv(file_path):
    """
    统计 CSV 文件中 label 列值为 0、1、2、3 的数量
    :param file_path: CSV 文件路径
    :return: 包含 label 数量的字典
    """
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    label = int(row.get('label'))
                    if label in label_counts:
                        label_counts[label] += 1
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        logging.error(f"Error occurred while reading file {file_path}: {e}")
    return label_counts

def write_results_to_csv(results, output_path):
    """
    将统计结果写入新的 CSV 文件
    :param results: 统计结果列表
    :param output_path: 输出文件路径
    """
    fieldnames = ['SBID', 0, 1, 2, 3]
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logging.info(f"Results have been successfully written to {output_path}")
    except Exception as e:
        logging.error(f"Error occurred while writing to file {output_path}: {e}")

def main():
    # Define the base path
    base_path = '/home/ydai240628/analysis_hetu/file/select_match_res/'
    output_path = '/home/ydai240628/analysis_hetu/file/select_match/label_count_summary_res.csv'

    # Initialize the result list
    results = []

    # Iterate through all CSV files in the directory
    for filename in os.listdir(base_path):
        if filename.endswith('.csv'):
            # Extract the XXX number name
            sbid = os.path.splitext(filename)[0]

            # Count the labels
            file_path = os.path.join(base_path, filename)
            label_counts = count_labels_in_csv(file_path)

            # Build the result row
            result_row = {'SBID': sbid}
            result_row.update(label_counts)
            results.append(result_row)

    # Write the results to a new CSV file
    write_results_to_csv(results, output_path)

if __name__ == "__main__":
    main()
    