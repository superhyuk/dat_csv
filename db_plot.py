import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import psycopg2
import numpy as np
from colorcet import fire, rainbow, coolwarm
from datetime import datetime
import os

# DB 연결
print("PostgreSQL 연결 중...")
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="pdm_db",
    user="pdm_user",
    password="pdm_password"
)

# 머신 ID와 날짜 범위 설정
machine_id = 'HOTCHAMBER_M2'  # 또는 'CURINGOVEN_M1'
start_date = '2025-04-01'
end_date = '2025-07-30'

# 전체 데이터 개수 확인
count_query = f"""
SELECT COUNT(*) 
FROM normal_acc_data 
WHERE machine_id = '{machine_id}'
AND time >= '{start_date}'
AND time < '{end_date}'
"""

cur = conn.cursor()
cur.execute(count_query)
total_count = cur.fetchone()[0]
print(f"총 데이터 개수: {total_count:,}")

# 데이터 로드 (청크 단위로)
print("\n데이터 로딩 중...")
chunk_size = 500000
chunks = []

for offset in range(0, total_count, chunk_size):
    query = f"""
    SELECT 
        EXTRACT(EPOCH FROM time) as time_epoch,
        x, y, z
    FROM normal_acc_data 
    WHERE machine_id = '{machine_id}'
    AND time >= '{start_date}'
    AND time < '{end_date}'
    ORDER BY time
    LIMIT {chunk_size} OFFSET {offset}
    """
    
    chunk = pd.read_sql(query, conn)
    chunks.append(chunk)
    print(f"  로드됨: {offset + len(chunk):,} / {total_count:,}")

# 데이터 합치기
df = pd.concat(chunks, ignore_index=True)
print(f"\n전체 로드 완료: {len(df):,} 행")

# 시간을 0부터 시작하는 인덱스로 변환 (시각화 성능 향상)
df['time_index'] = range(len(df))

conn.close()

# Datashader 설정
print("\n그래프 생성 중...")

# 캔버스 크기 설정 (해상도)
width = 2000
height = 600

# 1. 개별 축 시각화
for axis in ['x', 'y', 'z']:
    print(f"\n{axis.upper()}축 렌더링...")
    
    # 캔버스 생성
    cvs = ds.Canvas(plot_width=width, plot_height=height)
    
    # 라인 집계
    agg = cvs.line(df, x='time_index', y=axis)
    
    # 색상 적용
    img = tf.shade(agg, cmap=fire if axis=='x' else (coolwarm if axis=='y' else rainbow))
    
    # 배경 설정
    img = tf.set_background(img, 'white')
    
    # 이미지로 저장
    filename = f'acc_{axis}_axis_{machine_id}.png'
    img.to_pil().save(filename)
    print(f"  저장됨: {filename}")

# 2. 3축 통합 시각화
print("\n3축 통합 그래프 생성...")

# 각 축을 다른 색상으로
cvs = ds.Canvas(plot_width=width, plot_height=height*2)

# X축 - 파란색
agg_x = cvs.line(df, x='time_index', y='x')
img_x = tf.shade(agg_x, cmap=['lightblue', 'darkblue'])

# Y축 - 녹색  
agg_y = cvs.line(df, x='time_index', y='y')
img_y = tf.shade(agg_y, cmap=['lightgreen', 'darkgreen'])

# Z축 - 빨간색
agg_z = cvs.line(df, x='time_index', y='z')
img_z = tf.shade(agg_z, cmap=['pink', 'darkred'])

# 이미지 합성
from datashader.utils import export_image
export_image(img_x, filename='acc_all_axes', background='white')

print("\n✅ 모든 그래프 생성 완료!")

# 3. 인터랙티브 버전 (선택사항)
print("\n인터랙티브 HTML 생성 중...")

from bokeh.plotting import figure, output_file, save
from bokeh.models import DatetimeTickFormatter

# 시간 축 복원
df['time'] = pd.to_datetime(df['time_epoch'], unit='s')

# 샘플링 (인터랙티브용)
sample_size = min(100000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).sort_values('time')

p = figure(
    width=1400, 
    height=600,
    x_axis_type='datetime',
    title=f'{machine_id} ACC Data ({total_count:,} points, showing {sample_size:,})',
    tools='pan,wheel_zoom,box_zoom,reset,save'
)

p.line(df_sample['time'], df_sample['x'], color='blue', alpha=0.7, legend_label='X-axis')
p.line(df_sample['time'], df_sample['y'], color='green', alpha=0.7, legend_label='Y-axis')
p.line(df_sample['time'], df_sample['z'], color='red', alpha=0.7, legend_label='Z-axis')

p.legend.location = "top_right"
p.legend.click_policy = "hide"

output_file(f"acc_interactive_{machine_id}.html")
save(p)
print(f"  저장됨: acc_interactive_{machine_id}.html")

# 4. MIC 데이터도 처리
print("\n\nMIC 데이터 처리...")

# MIC 데이터 개수 확인
cur = conn.cursor()
cur.execute(f"""
SELECT COUNT(*) 
FROM normal_mic_data 
WHERE machine_id = '{machine_id}'
AND time >= '{start_date}'
AND time < '{end_date}'
""")
mic_count = cur.fetchone()[0]
print(f"MIC 데이터 개수: {mic_count:,}")

if mic_count > 0:
    # MIC 데이터 로드
    mic_chunks = []
    for offset in range(0, mic_count, chunk_size):
        query = f"""
        SELECT 
            EXTRACT(EPOCH FROM time) as time_epoch,
            mic_value
        FROM normal_mic_data 
        WHERE machine_id = '{machine_id}'
        AND time >= '{start_date}'
        AND time < '{end_date}'
        ORDER BY time
        LIMIT {chunk_size} OFFSET {offset}
        """
        
        chunk = pd.read_sql(query, conn)
        mic_chunks.append(chunk)
    
    df_mic = pd.concat(mic_chunks, ignore_index=True)
    df_mic['time_index'] = range(len(df_mic))
    
    # MIC 시각화
    cvs_mic = ds.Canvas(plot_width=width, plot_height=height)
    agg_mic = cvs_mic.line(df_mic, x='time_index', y='mic_value')
    img_mic = tf.shade(agg_mic, cmap=fire)
    img_mic = tf.set_background(img_mic, 'white')
    img_mic.to_pil().save(f'mic_{machine_id}.png')
    print(f"  MIC 그래프 저장됨: mic_{machine_id}.png")

print("\n🎉 모든 작업 완료!")
print(f"\n생성된 파일:")
print(f"  - acc_x_axis_{machine_id}.png")
print(f"  - acc_y_axis_{machine_id}.png")
print(f"  - acc_z_axis_{machine_id}.png")
print(f"  - acc_interactive_{machine_id}.html")
if mic_count > 0:
    print(f"  - mic_{machine_id}.png")