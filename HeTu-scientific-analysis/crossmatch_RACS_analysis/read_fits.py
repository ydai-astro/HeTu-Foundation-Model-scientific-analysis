import os
import csv
from astropy.io import fits

# 定义 FITS 文件所在的文件夹路径
fits_folder = '/groups/hetu_ai/home/share/racs-mid-images'

# 定义输出的 CSV 文件路径
csv_file = 'fits_output.csv'

# 打开 CSV 文件以写入数据
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入 CSV 文件的表头
    writer.writerow(['FITS File', 'SBID', 'CRVAL1', 'CRVAL2'])

    # 遍历文件夹中的所有文件
    for filename in os.listdir(fits_folder):
        # 检查文件是否为 FITS 文件
        if filename.endswith('.fits'):
            fits_path = os.path.join(fits_folder, filename)
            try:
                # 打开 FITS 文件
                with fits.open(fits_path) as hdul:
                    # 获取头部信息
                    header = hdul[0].header
                    # 获取 SBID、CRVAL1 和 CRVAL2 的值
                    sbid = header.get('SBID', 'N/A')
                    crval1 = header.get('CRVAL1', 'N/A')
                    crval2 = header.get('CRVAL2', 'N/A')
                    # 将信息写入 CSV 文件
                    writer.writerow([filename, sbid, crval1, crval2])
            except Exception as e:
                print(f"无法读取文件 {filename}: {e}")
