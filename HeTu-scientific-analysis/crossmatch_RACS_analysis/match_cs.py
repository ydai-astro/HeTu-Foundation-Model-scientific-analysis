#!/usr/bin/env python
import os
import re
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# 设置目录（请根据实际情况修改）
dirA = '/groups/hetu_ai/home/share/racs-mid-csv/'   # 存放 A 星表的目录
dirB = '/groups/hetu_ai/home/share/HeTu/xzj_code/rst/output_resnet/csv/'   # 存放 B 星表的目录
output_dir1 = 'output_resnet/csv'
output_dir2 = 'output_resnet/txt'

os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)

# 用于提取文件名中的 SB_数字部分，支持 "SB33098" 或 "SB_33098" 格式
pattern = re.compile(r"(SB[_]?(\d+))", re.IGNORECASE)

# 遍历 A 目录中的 CSV 文件
for fileA in os.listdir(dirA):
    if not fileA.lower().endswith('.csv'):
        continue
    matchA = pattern.search(fileA)
    if not matchA:
        continue
    sb_part = matchA.group(1)  # 如 "SB_33098" 或 "SB33098"
    
    # 在 B 目录中查找包含相同 sb_part 的文件（不区分大小写）
    candidates = [f for f in os.listdir(dirB) 
                  if f.lower().endswith('.csv') and sb_part.lower() in f.lower()]
    if not candidates:
        print(f"No matching B file found for {fileA}.")
        continue
    # 这里取第一个匹配的文件（可根据需要调整策略）
    fileB = candidates[0]
    
    # 构造完整路径
    pathA = os.path.join(dirA, fileA)
    pathB = os.path.join(dirB, fileB)
    
    print(f"Processing: A file '{fileA}' and B file '{fileB}'")
    
    # 读取数据
    dfA = pd.read_csv(pathA)
    dfB = pd.read_csv(pathB)
    
    # 对 B 星表筛选，只保留 labels == 1 的记录
    dfB0 = dfB[dfB['labels'] == 1].reset_index(drop=True)
    
    # 根据 A 表的 col_component_id 与 B 表的 component_id 进行合并
    merged = pd.merge(dfA, dfB0, left_on='col_component_id', right_on='component_id', 
                      how='inner', suffixes=('_A', '_B'))
    
    # 构造 SkyCoord 对象，计算合并后两表的坐标角距离
    catA = SkyCoord(ra=merged['col_ra_deg_cont'].values * u.deg,
                    dec=merged['col_dec_deg_cont'].values * u.deg)
    catB = SkyCoord(ra=merged['RA'].values * u.deg,
                    dec=merged['Dec'].values * u.deg)
    d2d = catA.separation(catB)
    
    # 设置匹配阈值为 40 角秒，并过滤出匹配成功的记录
    match_threshold = 40 * u.arcsec
    mask = d2d < match_threshold
    matched = merged[mask].copy()
    matched['Separation_arcsec'] = d2d[mask].arcsec
    
    # 保存匹配结果到 CSV 文件，文件名格式为 matched_catalog_<SB_数字>.csv
    out_matched = os.path.join(output_dir1, f"matched_catalog_{sb_part}.csv")
    matched.to_csv(out_matched, index=False)
    print(f"Matched results saved to '{out_matched}'.")
    
    # 计算匹配到的 A 星表源占整个 A 星表的比例
    total_A = len(dfA)
    matched_count = len(matched)
    match_fraction = matched_count / total_A if total_A > 0 else 0
    
    out_ratio = os.path.join(output_dir2, f"match_ratio_{sb_part}.txt")
    with open(out_ratio, 'w') as f:
        f.write("Matched A sources / Total A sources: {} / {} = {:.2%}\n"
                .format(matched_count, total_A, match_fraction))
    print(matched_count, total_A, match_fraction)
