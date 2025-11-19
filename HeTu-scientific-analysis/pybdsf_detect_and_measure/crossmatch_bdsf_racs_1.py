import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# 读入数据
df1 = pd.read_csv("/home/ydai240628/analysis_hetu/bdsf_out/20376/20376_srl.csv", comment='#', sep=',')
df2 = pd.read_csv("/home/ydai240628/analysis_hetu/RACS-mid-final-cateloge-primary/output.csv", comment='#', sep=',')
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
print("df1 列名：", df1.columns.tolist())
print("df2 列名：", df2.columns.tolist())


# 转换为天球坐标
coords1 = SkyCoord(ra=df1['RA'].values * u.deg,
                   dec=df1['DEC'].values * u.deg)
coords2 = SkyCoord(ra=df2['RA'].values * u.deg,
                   dec=df2['Dec'].values * u.deg)

# 在 40″ 内搜索匹配
idx1, idx2, d2d, d3d = coords1.search_around_sky(coords2, 20 * u.arcsec)

# 所有匹配结果
matches = pd.DataFrame({
    'df1_index': idx2,
    'df2_index': idx1,
    'sep_arcsec': d2d.arcsec
})

# 只保留每个 df1 最近的匹配
nearest = matches.loc[matches.groupby('df1_index')['sep_arcsec'].idxmin()].reset_index(drop=True)

# 合并匹配结果，只保留匹配到的 df2 数据
result = df1.loc[nearest['df1_index']].reset_index(drop=True)
result = result.join(
    df2.add_prefix('df3_').iloc[nearest['df2_index']].reset_index(drop=True)
)

# 加上匹配的角距离和 Isl_id 标识
result['sep_arcsec'] = nearest['sep_arcsec']

# 保存结果
out_path = "/home/ydai240628/analysis_hetu/file/match_bdsf_hetu/matched_bdsf_racs_srl_20arcsec.csv"
result.to_csv(out_path, index=False)
print(f"匹配结果已保存为 {out_path}")
