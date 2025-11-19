import os
import re
import csv

def extract_subdirectory_proportion(file_path):
    subdirectory, proportion = None, None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            subdir_match = re.search(r'Subdirectory:\s*(\S+)', line)
            proportion_match = re.search(r'Proportion:\s*([\d\.]+)', line)
            
            if subdir_match:
                subdirectory = subdir_match.group(1)
            if proportion_match:
                proportion = float(proportion_match.group(1))
            
            if subdirectory and proportion:
                break
    return subdirectory, proportion

def read_txt_files_from_folder(folder_path):
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            subdirectory, proportion = extract_subdirectory_proportion(file_path)
            results.append((file_name, subdirectory, proportion))
    return results

def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["File Name", "Subdirectory", "Proportion"])
        csv_writer.writerows(data)

if __name__ == "__main__":
    folder_path = "/groups/hetu_ai/home/share/HeTu/xzj_code/rst/output_resnet/txt/"  # 替换成你的文件夹路径
    output_csv = "./anyl_resnet.csv"
    extracted_data = read_txt_files_from_folder(folder_path)
    
    write_to_csv(extracted_data, output_csv)
    
    for file_name, subdirectory, proportion in extracted_data:
        print(f"File: {file_name}, Subdirectory: {subdirectory}, Proportion: {proportion}")
