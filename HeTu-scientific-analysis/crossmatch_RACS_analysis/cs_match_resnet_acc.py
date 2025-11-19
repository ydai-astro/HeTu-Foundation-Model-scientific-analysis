import os
import pandas as pd

# 设置包含 txt 文件的目录，请根据需要修改路径
txt_dir = "/home/ydai240628/analysis_hetu/code/output_internimage_0722/txt/"  
# 输出 CSV 文件路径
output_csv = "cs_match_internimage_0722.csv"

data = []

# 遍历目录中的所有文件
for filename in os.listdir(txt_dir):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(txt_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            # 读取每个 txt 文件中的第一行内容，并去除首尾空白字符
            content = f.readline().strip()
            data.append((filename, content))

# 将数据转换为 DataFrame，设置列名
df = pd.DataFrame(data, columns=["component_id", "Content"])
# 保存为 CSV 文件，不保存行索引
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"CSV file saved to {output_csv}")
