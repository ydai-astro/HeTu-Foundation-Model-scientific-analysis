import json
import os
import csv


def is_overlapping(box1, box2):
    """
    判断两个边界框是否重叠
    :param box1: 第一个边界框 [x1, y1, x2, y2]
    :param box2: 第二个边界框 [x1, y1, x2, y2]
    :return: 如果重叠返回 True，否则返回 False
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)


def process_json_data(json_data):
    labels = json_data["labels"]
    scores = json_data["scores"]
    bboxes = json_data["bboxes"]
    masks = json_data["masks"]

    final_labels = []
    final_scores = []
    final_bboxes = []
    final_masks = []

    processed_indices = set()

    for i in range(len(bboxes)):
        if i in processed_indices:
            continue
        current_box = bboxes[i]
        current_score = scores[i]
        max_score_index = i
        max_score = current_score

        for j in range(i + 1, len(bboxes)):
            if j in processed_indices:
                continue
            other_box = bboxes[j]
            other_score = scores[j]

            if is_overlapping(current_box, other_box):
                if other_score > max_score:
                    max_score = other_score
                    max_score_index = j
                processed_indices.add(j)

        final_labels.append(labels[max_score_index])
        final_scores.append(scores[max_score_index])
        final_bboxes.append(bboxes[max_score_index])
        final_masks.append(masks[max_score_index])
        processed_indices.add(max_score_index)

    result = {
        "labels": final_labels,
        "scores": final_scores,
        "bboxes": final_bboxes,
        "masks": final_masks
    }
    return result


# 指定根目录，这里需要你修改为实际存放 JSON 文件的根目录
root_directory = '/groups/hetu_ai/home/share/HeTu/pjlab/AI4Astronomy_zhuanyi/output_resnet/'
# 指定输出路径，这里需要你修改为想要生成 CSV 文件的目标路径
output_path = '/home/ydai240628/analysis_hetu/file/only_label/output_resnet'

# 确保输出路径存在，如果不存在则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 遍历根目录下的所有文件夹
for root, dirs, files in os.walk(root_directory):
    if not any(file.endswith('.json') for file in files):
        continue

    # 获取上一级文件夹名
    parent_folder_name = os.path.basename(os.path.dirname(root))
    
    # 创建或打开CSV文件
    csv_file = os.path.join(output_path, f"{parent_folder_name}.csv")
    csvfile = open(csv_file, 'w', newline='')
    fieldnames = ['component_id', 'label', 'score', 'bbox', 'counts']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # 遍历当前文件夹下的所有文件
    for filename in files:
        if filename.endswith('.json'):
            file_path = os.path.join(root, filename)
            try:
                # 打开文件并读取 JSON 数据
                with open(file_path, 'r') as file:
                    json_str = file.read()

                # 解析 JSON 数据
                data = json.loads(json_str)
                print(f"Successfully parsed {file_path}, data length: {len(data.get('labels', []))}")

                # 处理数据
                result = process_json_data(data)
                print(f"Processed result for {file_path}: {result}")  # 打印处理后的结果

                # 写入处理后的结果到 CSV 文件，包含counts信息
                for label, score, bbox, mask in zip(result["labels"], result["scores"], result["bboxes"], result["masks"]):
                    counts = mask.get("counts", "")  # 获取counts，如果不存在则为空字符串
                    writer.writerow({
                        'component_id': filename,
                        'label': label,
                        'score': score,
                        'bbox': str(bbox),
                        'counts': counts
                    })
                csvfile.flush()  # 立即刷新缓冲区

            except FileNotFoundError:
                print(f"{file_path} not found")
            except json.JSONDecodeError:
                print(f"{file_path} not effective json")
            except KeyError as e:
                print(f"KeyError in {file_path}: {e}")
                print(f"Available keys: {list(data.keys())}")
            except Exception as e:
                print(f"Unexpected error in {file_path}: {e}")
    
    # 关闭当前文件夹对应的CSV文件
    csvfile.close()
