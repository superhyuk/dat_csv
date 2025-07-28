# s3_to_timescaledb_gui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime, timedelta
import threading
import boto3
import psycopg2
import numpy as np
from io import BytesIO
import os
from dotenv import load_dotenv
import time
import struct

from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from psycopg2.extras import execute_values

class S3ToTimescaleDBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("S3 DAT to TimescaleDB Converter")
        self.root.geometry("900x800")
        
        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        
        # 변수 초기화
        self.s3_client = None
        self.db_conn = None
        self.normal_periods = []
        self.anomaly_periods = []
        self.is_processing = False
        
        # 배치 처리 설정
        self.batch_size = 10000  # 한 번에 삽입할 레코드 수
        self.file_batch_size = 50  # 한 번에 처리할 파일 수
        self.commit_interval = 100  # 커밋 간격 (파일 수)
        self.max_workers = 5  # 동시 처리 워커 수
        
        # 통계 정보
        self.stats = {'processed_files': 0, 'total_records': 0, 'start_time': None}
        
        # UI 생성
        self.create_widgets()
        
        # 환경 변수 로드
        load_dotenv()
        
    def create_widgets(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 1. AWS/DB 설정 섹션
        self.create_config_section(main_frame)
        
        # 2. 머신 및 센서 선택
        self.create_machine_section(main_frame)
        
        # 3. 기간 설정 섹션
        self.create_period_section(main_frame)
        
        # 4. 실행 버튼
        self.create_action_section(main_frame)
        
        # 5. 진행 상황 및 로그
        self.create_progress_section(main_frame)
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def create_config_section(self, parent):
        # 설정 프레임
        config_frame = ttk.LabelFrame(parent, text="설정", padding="5")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # .env 파일 로드 섹션
        env_frame = ttk.Frame(config_frame)
        env_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.env_label = ttk.Label(env_frame, text="환경 파일이 로드되지 않았습니다.", foreground="red")
        self.env_label.grid(row=0, column=0, padx=5)
        
        ttk.Button(env_frame, text=".env 파일 선택", command=self.load_env_file).grid(row=0, column=1, padx=5)
        ttk.Button(env_frame, text="새로고침", command=self.refresh_env_values).grid(row=0, column=2, padx=5)
        
        # 구분선
        ttk.Separator(config_frame, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # AWS 설정
        ttk.Label(config_frame, text="AWS Access Key:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.aws_key_var = tk.StringVar()
        self.aws_key_entry = ttk.Entry(config_frame, textvariable=self.aws_key_var, width=50)
        self.aws_key_entry.grid(row=2, column=1, padx=5)
        
        ttk.Label(config_frame, text="AWS Secret Key:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.aws_secret_var = tk.StringVar()
        self.aws_secret_entry = ttk.Entry(config_frame, textvariable=self.aws_secret_var, width=50, show="*")
        self.aws_secret_entry.grid(row=3, column=1, padx=5)
        
        ttk.Label(config_frame, text="AWS Region:").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.aws_region_var = tk.StringVar(value='ap-northeast-2')
        self.aws_region_entry = ttk.Entry(config_frame, textvariable=self.aws_region_var, width=50)
        self.aws_region_entry.grid(row=4, column=1, padx=5)
        
        ttk.Label(config_frame, text="S3 Bucket:").grid(row=5, column=0, sticky=tk.W, padx=5)
        self.bucket_var = tk.StringVar(value='vtnnbl')
        self.bucket_entry = ttk.Entry(config_frame, textvariable=self.bucket_var, width=50)
        self.bucket_entry.grid(row=5, column=1, padx=5)
        
        # 연결 테스트 버튼
        ttk.Button(config_frame, text="연결 테스트", command=self.test_connections).grid(row=6, column=0, columnspan=2, pady=10)
        
    def create_machine_section(self, parent):
        # 머신/센서 선택 프레임
        machine_frame = ttk.LabelFrame(parent, text="머신 및 센서 선택", padding="5")
        machine_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 머신 선택
        ttk.Label(machine_frame, text="머신:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.machine_var = tk.StringVar(value="CURINGOVEN_M1")
        machine_combo = ttk.Combobox(machine_frame, textvariable=self.machine_var, 
                                    values=["CURINGOVEN_M1", "HOTCHAMBER_M2"], 
                                    state="readonly", width=20)
        machine_combo.grid(row=0, column=1, padx=5)
        
        # 센서 선택
        ttk.Label(machine_frame, text="센서:").grid(row=0, column=2, sticky=tk.W, padx=20)
        self.sensor_acc = tk.BooleanVar(value=True)
        self.sensor_mic = tk.BooleanVar(value=True)
        ttk.Checkbutton(machine_frame, text="ACC", variable=self.sensor_acc).grid(row=0, column=3, padx=5)
        ttk.Checkbutton(machine_frame, text="MIC", variable=self.sensor_mic).grid(row=0, column=4, padx=5)
        
        # 성능 설정 추가
        ttk.Label(machine_frame, text="배치 크기:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.batch_size_var = tk.IntVar(value=10000)
        batch_spinbox = ttk.Spinbox(machine_frame, from_=1000, to=50000, increment=1000, 
                                   textvariable=self.batch_size_var, width=10)
        batch_spinbox.grid(row=1, column=1, padx=5)
        
        ttk.Label(machine_frame, text="동시 처리:").grid(row=1, column=2, sticky=tk.W, padx=20)
        self.workers_var = tk.IntVar(value=5)
        workers_spinbox = ttk.Spinbox(machine_frame, from_=1, to=20, increment=1,
                                     textvariable=self.workers_var, width=10)
        workers_spinbox.grid(row=1, column=3, padx=5)
        
        ttk.Label(machine_frame, text="파일 배치:").grid(row=1, column=4, sticky=tk.W, padx=5)
        self.file_batch_var = tk.IntVar(value=50)
        file_batch_spinbox = ttk.Spinbox(machine_frame, from_=10, to=200, increment=10,
                                        textvariable=self.file_batch_var, width=10)
        file_batch_spinbox.grid(row=1, column=5, padx=5)
        
    def create_period_section(self, parent):
        # 기간 설정 프레임
        period_frame = ttk.LabelFrame(parent, text="기간 설정", padding="5")
        period_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 정상 기간 프레임
        normal_frame = ttk.Frame(period_frame)
        normal_frame.grid(row=0, column=0, padx=10, sticky=(tk.W, tk.E))
        
        ttk.Label(normal_frame, text="정상 기간:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=4, sticky=tk.W)
        
        # 정상 기간 입력
        ttk.Label(normal_frame, text="시작일:").grid(row=1, column=0, sticky=tk.W)
        self.normal_start = ttk.Entry(normal_frame, width=12)
        self.normal_start.grid(row=1, column=1, padx=5)
        self.normal_start.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Label(normal_frame, text="종료일:").grid(row=1, column=2, sticky=tk.W)
        self.normal_end = ttk.Entry(normal_frame, width=12)
        self.normal_end.grid(row=1, column=3, padx=5)
        self.normal_end.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Button(normal_frame, text="추가", command=self.add_normal_period).grid(row=1, column=4, padx=5)
        
        # 정상 기간 목록
        self.normal_listbox = tk.Listbox(normal_frame, height=4, width=50)
        self.normal_listbox.grid(row=2, column=0, columnspan=5, pady=5)
        ttk.Button(normal_frame, text="선택 삭제", command=self.delete_normal_period).grid(row=3, column=0, columnspan=5)
        
        # 비정상 기간 프레임
        anomaly_frame = ttk.Frame(period_frame)
        anomaly_frame.grid(row=0, column=1, padx=10, sticky=(tk.W, tk.E))
        
        ttk.Label(anomaly_frame, text="비정상 기간:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=4, sticky=tk.W)
        
        # 비정상 기간 입력
        ttk.Label(anomaly_frame, text="시작일:").grid(row=1, column=0, sticky=tk.W)
        self.anomaly_start = ttk.Entry(anomaly_frame, width=12)
        self.anomaly_start.grid(row=1, column=1, padx=5)
        self.anomaly_start.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Label(anomaly_frame, text="종료일:").grid(row=1, column=2, sticky=tk.W)
        self.anomaly_end = ttk.Entry(anomaly_frame, width=12)
        self.anomaly_end.grid(row=1, column=3, padx=5)
        self.anomaly_end.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Button(anomaly_frame, text="추가", command=self.add_anomaly_period).grid(row=1, column=4, padx=5)
        
        # 비정상 기간 목록
        self.anomaly_listbox = tk.Listbox(anomaly_frame, height=4, width=50)
        self.anomaly_listbox.grid(row=2, column=0, columnspan=5, pady=5)
        ttk.Button(anomaly_frame, text="선택 삭제", command=self.delete_anomaly_period).grid(row=3, column=0, columnspan=5)
        
    def create_action_section(self, parent):
        # 실행 버튼 프레임
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.process_btn = ttk.Button(action_frame, text="처리 시작", command=self.start_processing, 
                                     style='Accent.TButton', padding=(20, 10))
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(action_frame, text="중지", command=self.stop_processing, 
                                  state='disabled', padding=(20, 10))
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
    def create_progress_section(self, parent):
        # 진행 상황 프레임
        progress_frame = ttk.LabelFrame(parent, text="진행 상황", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 진행률 바
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 상태 레이블
        self.status_label = ttk.Label(progress_frame, text="대기 중...")
        self.status_label.grid(row=1, column=0, sticky=tk.W, padx=5)
        
        # 통계 정보 레이블 추가
        self.stats_label = ttk.Label(progress_frame, text="")
        self.stats_label.grid(row=2, column=0, sticky=tk.W, padx=5)
        
        # 예상 시간 레이블
        self.eta_label = ttk.Label(progress_frame, text="")
        self.eta_label.grid(row=3, column=0, sticky=tk.W, padx=5)
        
        # 로그 텍스트
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=10, width=80)
        self.log_text.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        progress_frame.rowconfigure(4, weight=1)
        
    def test_connections(self):
        """AWS S3 및 TimescaleDB 연결 테스트"""
        try:
            # S3 연결 테스트
            self.s3_client = boto3.client('s3',
                aws_access_key_id=self.aws_key_var.get(),
                aws_secret_access_key=self.aws_secret_var.get(),
                region_name=self.aws_region_var.get()
            )
            
            # 버킷 리스트 확인
            buckets = self.s3_client.list_buckets()
            self.log(f"S3 연결 성공! {len(buckets['Buckets'])}개 버킷 발견")
            
            # TimescaleDB 연결 테스트
            self.db_conn = psycopg2.connect(
                host="localhost",
                port="5432",
                database="pdm_db",
                user="pdm_user",
                password="pdm_password"
            )
            self.log("TimescaleDB 연결 성공!")
            
            # TimescaleDB extension 확인
            cur = self.db_conn.cursor()
            cur.execute("SELECT default_version FROM pg_available_extensions WHERE name = 'timescaledb';")
            result = cur.fetchone()
            if result:
                self.log(f"TimescaleDB 버전: {result[0]}")
            cur.close()
            
            messagebox.showinfo("성공", "S3 및 TimescaleDB 연결 성공!")
            
        except Exception as e:
            self.log(f"연결 오류: {str(e)}")
            messagebox.showerror("오류", f"연결 실패: {str(e)}")
    
    def load_env_file(self):
        """환경 파일 선택 및 로드"""
        from tkinter import filedialog
        
        initial_dir = os.getcwd()
        env_path = filedialog.askopenfilename(
            title="환경 변수 파일 선택",
            initialdir=initial_dir,
            filetypes=[("ENV files", "*.env"), ("All files", "*.*")]
        )
        
        if env_path:
            try:
                # .env 파일 로드
                load_dotenv(env_path, override=True)
                
                # 로드된 값으로 UI 업데이트
                self.refresh_env_values()
                
                self.env_label.config(text=f"로드됨: {os.path.basename(env_path)}", foreground="green")
                self.log(f"환경 파일 로드 완료: {env_path}")
                
                # 로드된 값 확인
                self.log("로드된 환경 변수:")
                self.log(f"  AWS_ACCESS_KEY_ID: {'*' * 10 if os.getenv('AWS_ACCESS_KEY_ID') else '없음'}")
                self.log(f"  AWS_SECRET_ACCESS_KEY: {'*' * 10 if os.getenv('AWS_SECRET_ACCESS_KEY') else '없음'}")
                self.log(f"  AWS_REGION: {os.getenv('AWS_REGION', '없음')}")
                self.log(f"  S3_BUCKET: {os.getenv('S3_BUCKET', '없음')}")
                
            except Exception as e:
                messagebox.showerror("오류", f".env 파일 로드 중 오류 발생: {str(e)}")
                self.env_label.config(text="환경 파일 로드 실패", foreground="red")
                self.log(f"환경 파일 로드 오류: {str(e)}")

    def refresh_env_values(self):
        """환경 변수에서 값 다시 읽기"""
        self.aws_key_var.set(os.getenv('AWS_ACCESS_KEY_ID', ''))
        self.aws_secret_var.set(os.getenv('AWS_SECRET_ACCESS_KEY', ''))
        self.aws_region_var.set(os.getenv('AWS_REGION', 'ap-northeast-2'))
        self.bucket_var.set(os.getenv('S3_BUCKET', 'vtnnbl'))
        self.log("환경 변수 값 새로고침 완료")
    
    def add_normal_period(self):
        """정상 기간 추가"""
        try:
            start = datetime.strptime(self.normal_start.get(), "%Y-%m-%d")
            end = datetime.strptime(self.normal_end.get(), "%Y-%m-%d")
            if start > end:
                messagebox.showerror("오류", "시작일이 종료일보다 늦습니다.")
                return
            
            period = {"start": start, "end": end}
            self.normal_periods.append(period)
            self.normal_listbox.insert(tk.END, f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
            self.log(f"정상 기간 추가: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
            
        except ValueError:
            messagebox.showerror("오류", "날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)")
    
    def add_anomaly_period(self):
        """비정상 기간 추가"""
        try:
            start = datetime.strptime(self.anomaly_start.get(), "%Y-%m-%d")
            end = datetime.strptime(self.anomaly_end.get(), "%Y-%m-%d")
            if start > end:
                messagebox.showerror("오류", "시작일이 종료일보다 늦습니다.")
                return
            
            period = {"start": start, "end": end}
            self.anomaly_periods.append(period)
            self.anomaly_listbox.insert(tk.END, f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
            self.log(f"비정상 기간 추가: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
            
        except ValueError:
            messagebox.showerror("오류", "날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)")
    
    def delete_normal_period(self):
        """선택된 정상 기간 삭제"""
        selection = self.normal_listbox.curselection()
        if selection:
            index = selection[0]
            self.normal_listbox.delete(index)
            del self.normal_periods[index]
            self.log("정상 기간 삭제됨")
    
    def delete_anomaly_period(self):
        """선택된 비정상 기간 삭제"""
        selection = self.anomaly_listbox.curselection()
        if selection:
            index = selection[0]
            self.anomaly_listbox.delete(index)
            del self.anomaly_periods[index]
            self.log("비정상 기간 삭제됨")
    
    def log(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def parse_filename_date(self, filename):
        """파일명에서 날짜 추출"""
        # 20250407_11_28_22_MP23ABS1_MIC.dat
        parts = filename.split('_')
        if len(parts) >= 4:
            date_str = parts[0]
            hour = int(parts[1])
            minute = int(parts[2])
            second = int(parts[3])
            
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            return datetime(year, month, day, hour, minute, second)
        return None
    
    def is_in_period(self, date, periods):
        """날짜가 지정된 기간들 중 하나에 포함되는지 확인"""
        # date가 datetime 객체인 경우 date()로 변환
        if isinstance(date, datetime):
            date_only = date.date()
        else:
            date_only = date
            
        for period in periods:
            # period의 start와 end도 date 객체로 변환
            start_date = period['start'].date() if isinstance(period['start'], datetime) else period['start']
            end_date = period['end'].date() if isinstance(period['end'], datetime) else period['end']
            
            if start_date <= date_only <= end_date:
                return True
        return False
    
    def create_tables(self):
        """TimescaleDB 테이블 생성"""
        cur = self.db_conn.cursor()
        
        # TimescaleDB extension 활성화
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # 정상 ACC 테이블
        cur.execute("""
            CREATE TABLE IF NOT EXISTS normal_acc_data (
                time TIMESTAMPTZ NOT NULL,
                machine_id TEXT NOT NULL,
                x DOUBLE PRECISION,
                y DOUBLE PRECISION,
                z DOUBLE PRECISION,
                filename TEXT
            );
        """)
        
        # 정상 MIC 테이블
        cur.execute("""
            CREATE TABLE IF NOT EXISTS normal_mic_data (
                time TIMESTAMPTZ NOT NULL,
                machine_id TEXT NOT NULL,
                mic_value INTEGER,
                filename TEXT
            );
        """)
        
        # 비정상 ACC 테이블
        cur.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_acc_data (
                time TIMESTAMPTZ NOT NULL,
                machine_id TEXT NOT NULL,
                x DOUBLE PRECISION,
                y DOUBLE PRECISION,
                z DOUBLE PRECISION,
                filename TEXT
            );
        """)
        
        # 비정상 MIC 테이블
        cur.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_mic_data (
                time TIMESTAMPTZ NOT NULL,
                machine_id TEXT NOT NULL,
                mic_value INTEGER,
                filename TEXT
            );
        """)
        
        # 하이퍼테이블로 변환
        tables = ['normal_acc_data', 'normal_mic_data', 'anomaly_acc_data', 'anomaly_mic_data']
        for table in tables:
            try:
                cur.execute(f"SELECT create_hypertable('{table}', 'time', if_not_exists => TRUE);")
                self.log(f"{table} 하이퍼테이블 생성/확인 완료")
                
                # 압축 설정
                cur.execute(f"""
                    ALTER TABLE {table} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'machine_id',
                        timescaledb.compress_orderby = 'time DESC'
                    );
                """)
                
            except Exception as e:
                self.log(f"{table} 하이퍼테이블 설정 중 오류 (이미 존재할 수 있음): {str(e)}")
        
        self.db_conn.commit()
        cur.close()
    
    def load_acc_dat_from_s3(self, bucket, key):
        """S3에서 ACC DAT 파일 읽기"""
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        data_stream = BytesIO(obj['Body'].read())
        
        # 파일 크기 확인
        file_size = data_stream.seek(0, 2)
        data_stream.seek(0)
        
        # 각 세트는 3000개 int16 (6000바이트) + 8바이트 스킵 = 6008바이트
        sets = file_size // 6008
        samples = sets * 1000
        
        data = np.empty((samples, 3), dtype=np.float64)
        total = 0
        
        for i in range(sets):
            # 3000개 int16 읽기
            chunk_bytes = data_stream.read(6000)
            if len(chunk_bytes) < 6000:
                break
                
            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
            chunk_data = chunk.reshape(-1, 3)
            
            csize = min(1000, samples - total)
            data[total:total+csize, :] = chunk_data[:csize, :].astype(np.float64) * 0.000488
            total += csize
            
            # 8바이트 스킵
            data_stream.read(8)
        
        return data[:total]
    
    def load_mic_dat_from_s3(self, bucket, key):
        """S3에서 MIC DAT 파일 읽기"""
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        data_stream = BytesIO(obj['Body'].read())
        
        # 파일 크기 확인
        file_size = data_stream.seek(0, 2)
        data_stream.seek(0)
        
        # 각 세트는 1000개 int16 (2000바이트) + 8바이트 스킵 = 2008바이트
        sets = file_size // 2008
        samples = sets * 1000
        
        data = np.empty(samples, dtype=np.int16)
        
        for i in range(sets):
            # 1000개 int16 읽기
            chunk_bytes = data_stream.read(2000)
            if len(chunk_bytes) < 2000:
                break
                
            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
            data[i*1000:(i+1)*1000] = chunk
            
            # 8바이트 스킵
            data_stream.read(8)
        
        return data[:i*1000+len(chunk)]
    
    def insert_acc_data_batch(self, table_name, batch_data):
        """ACC 데이터를 배치로 DB에 삽입"""
        cur = self.db_conn.cursor()
        
        try:
            # COPY를 사용한 대량 삽입을 위해 데이터 준비
            execute_values(
                cur,
                f"INSERT INTO {table_name} (time, machine_id, x, y, z, filename) VALUES %s",
                batch_data,
                template="(%s, %s, %s, %s, %s, %s)",
                page_size=1000
            )
            
            inserted = len(batch_data)
        except Exception as e:
            self.log(f"배치 삽입 오류: {str(e)}")
            self.db_conn.rollback()
            raise
        finally:
            cur.close()
        
        return inserted
    
    def insert_mic_data_batch(self, table_name, batch_data):
        """MIC 데이터를 배치로 DB에 삽입"""
        cur = self.db_conn.cursor()
        
        try:
            execute_values(
                cur,
                f"INSERT INTO {table_name} (time, machine_id, mic_value, filename) VALUES %s",
                batch_data,
                template="(%s, %s, %s, %s)",
                page_size=1000
            )
            
            inserted = len(batch_data)
        except Exception as e:
            self.log(f"배치 삽입 오류: {str(e)}")
            self.db_conn.rollback()
            raise
        finally:
            cur.close()
        
        return inserted
    
    def process_file_batch(self, file_batch, machine_id):
        """파일 배치를 처리하고 데이터를 준비"""
        acc_normal_batch = []
        acc_anomaly_batch = []
        mic_normal_batch = []
        mic_anomaly_batch = []
        
        bucket = self.bucket_var.get()
        
        for sensor, key in file_batch:
            if not self.is_processing:
                break
                
            filename = os.path.basename(key)
            file_date = self.parse_filename_date(filename)
            
            if not file_date:
                continue
            
            # 정상/비정상 판단
            is_normal = self.is_in_period(file_date.date(), self.normal_periods)
            is_anomaly = self.is_in_period(file_date.date(), self.anomaly_periods)
            
            if not is_normal and not is_anomaly:
                continue
            
            try:
                if sensor == 'acc':
                    data = self.load_acc_dat_from_s3(bucket, key)
                    sampling_rate = 1666.0
                    
                    # 데이터를 배치에 추가
                    for i in range(len(data)):
                        time_offset = i / sampling_rate
                        timestamp = file_date + timedelta(seconds=time_offset)
                        row = (timestamp, machine_id, float(data[i, 0]), float(data[i, 1]), float(data[i, 2]), filename)
                        
                        if is_normal:
                            acc_normal_batch.append(row)
                        else:
                            acc_anomaly_batch.append(row)
                            
                else:  # mic
                    data = self.load_mic_dat_from_s3(bucket, key)
                    sampling_rate = 8000.0
                    
                    # 데이터를 배치에 추가
                    for i in range(len(data)):
                        time_offset = i / sampling_rate
                        timestamp = file_date + timedelta(seconds=time_offset)
                        row = (timestamp, machine_id, int(data[i]), filename)
                        
                        if is_normal:
                            mic_normal_batch.append(row)
                        else:
                            mic_anomaly_batch.append(row)
                            
            except Exception as e:
                self.log(f"파일 처리 오류 ({filename}): {str(e)}")
                continue
        
        return {
            'acc_normal': acc_normal_batch,
            'acc_anomaly': acc_anomaly_batch,
            'mic_normal': mic_normal_batch,
            'mic_anomaly': mic_anomaly_batch
        }
    
    def compress_table_chunks(self, table_name):
        """테이블의 압축되지 않은 청크 압축"""
        cur = self.db_conn.cursor()
        
        try:
            # 압축되지 않은 청크 찾기
            cur.execute(f"""
                SELECT compress_chunk(i, if_not_compressed=>true)
                FROM show_chunks('{table_name}') i;
            """)
            
            compressed = cur.fetchall()
            if compressed:
                self.log(f"{table_name}: {len(compressed)}개 청크 압축 완료")
                
        except Exception as e:
            self.log(f"{table_name} 압축 중 오류: {str(e)}")
        
        self.db_conn.commit()
        cur.close()
    
    def process_s3_files(self):
        """S3 파일 처리 메인 로직"""
        try:
            # 성능 설정 업데이트
            self.batch_size = self.batch_size_var.get()
            self.max_workers = self.workers_var.get()
            self.file_batch_size = self.file_batch_var.get()
            
            bucket = self.bucket_var.get()  # 환경 변수나 입력값에서 가져오기
            machine_id = self.machine_var.get()
            
            # 처리할 센서 타입
            sensors = []
            if self.sensor_acc.get():
                sensors.append('acc')
            if self.sensor_mic.get():
                sensors.append('mic')
            
            if not sensors:
                messagebox.showerror("오류", "최소 하나의 센서를 선택해주세요.")
                return
            
            # 테이블 생성
            self.create_tables()
            
            # 전체 파일 목록 수집
            all_files = []
            for sensor in sensors:
                prefix = f"{machine_id}/raw_dat/{sensor}/"
                self.log(f"{sensor.upper()} 파일 목록 가져오는 중...")
                
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith('.dat'):
                                all_files.append((sensor, obj['Key']))
            
            self.log(f"총 {len(all_files)}개 파일 발견")
            
            # 통계 초기화
            self.stats['start_time'] = time.time()
            self.stats['processed_files'] = 0
            self.stats['total_records'] = 0
            
            # 파일을 배치로 나누기
            file_batches = [all_files[i:i+self.file_batch_size] 
                           for i in range(0, len(all_files), self.file_batch_size)]
            
            # ThreadPoolExecutor를 사용한 병렬 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for batch_idx, file_batch in enumerate(file_batches):
                    if not self.is_processing:
                        break
                    
                    future = executor.submit(self.process_file_batch, file_batch, machine_id)
                    futures.append((batch_idx, future))
                    
                # 결과 수집 및 DB 삽입
                for batch_idx, future in futures:
                    if not self.is_processing:
                        break
                        
                    try:
                        result = future.result(timeout=300)  # 5분 타임아웃
                        
                        # 배치 데이터 삽입
                        if result['acc_normal']:
                            self.insert_acc_data_batch('normal_acc_data', result['acc_normal'])
                            self.stats['total_records'] += len(result['acc_normal'])
                            
                        if result['acc_anomaly']:
                            self.insert_acc_data_batch('anomaly_acc_data', result['acc_anomaly'])
                            self.stats['total_records'] += len(result['acc_anomaly'])
                            
                        if result['mic_normal']:
                            self.insert_mic_data_batch('normal_mic_data', result['mic_normal'])
                            self.stats['total_records'] += len(result['mic_normal'])
                            
                        if result['mic_anomaly']:
                            self.insert_mic_data_batch('anomaly_mic_data', result['mic_anomaly'])
                            self.stats['total_records'] += len(result['mic_anomaly'])
                        
                        # 통계 업데이트
                        self.stats['processed_files'] += len(file_batch)
                        
                        # 주기적 커밋
                        if batch_idx % 10 == 0:  # 10개 배치마다 커밋
                            self.db_conn.commit()
                            
                        # 진행 상황 업데이트
                        progress = (self.stats['processed_files'] / len(all_files)) * 100
                        self.progress_var.set(progress)
                        
                        # 처리 속도 계산 및 예상 시간
                        elapsed = time.time() - self.stats['start_time']
                        if elapsed > 0:
                            files_per_sec = self.stats['processed_files'] / elapsed
                            records_per_sec = self.stats['total_records'] / elapsed
                            remaining_files = len(all_files) - self.stats['processed_files']
                            eta = remaining_files / files_per_sec if files_per_sec > 0 else 0
                            
                            self.status_label.config(
                                text=f"처리 중: 배치 {batch_idx+1}/{len(file_batches)} "
                                     f"({self.stats['processed_files']}/{len(all_files)} 파일)"
                            )
                            
                            self.stats_label.config(
                                text=f"속도: {files_per_sec:.1f} 파일/초, "
                                     f"{records_per_sec:.0f} 레코드/초"
                            )
                            
                            eta_hours = int(eta // 3600)
                            eta_minutes = int((eta % 3600) // 60)
                            self.eta_label.config(
                                text=f"예상 남은 시간: {eta_hours}시간 {eta_minutes}분"
                            )
                        
                        # 로그 (배치마다)
                        if batch_idx % 5 == 0:
                            self.log(f"배치 {batch_idx+1}/{len(file_batches)} 완료 - "
                                   f"총 {self.stats['processed_files']} 파일, "
                                   f"{self.stats['total_records']:,} 레코드 처리")
                        
                        # 주기적 압축 (20개 배치마다)
                        if batch_idx % 20 == 0 and batch_idx > 0:
                            self.log("청크 압축 중...")
                            for table_name in ['normal_acc_data', 'normal_mic_data', 
                                             'anomaly_acc_data', 'anomaly_mic_data']:
                                self.compress_table_chunks(table_name)
                                
                    except Exception as e:
                        self.log(f"배치 {batch_idx} 처리 오류: {str(e)}")
                        continue
            
            # 최종 커밋 및 압축
            self.db_conn.commit()
            self.log("최종 압축 중...")
            for table_name in ['normal_acc_data', 'normal_mic_data', 'anomaly_acc_data', 'anomaly_mic_data']:
                self.compress_table_chunks(table_name)
            
            # 처리 시간 계산
            total_time = time.time() - self.stats['start_time']
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            
            self.log(f"\n처리 완료!")
            self.log(f"총 처리 시간: {hours}시간 {minutes}분")
            self.log(f"총 파일: {self.stats['processed_files']:,}")
            self.log(f"총 레코드: {self.stats['total_records']:,}")
            
            messagebox.showinfo("완료", 
                f"처리 완료!\n\n"
                f"처리 시간: {hours}시간 {minutes}분\n"
                f"총 파일: {self.stats['processed_files']:,}\n"
                f"총 레코드: {self.stats['total_records']:,}")
            
        except Exception as e:
            self.log(f"처리 중 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"처리 중 오류 발생: {str(e)}")
        
        finally:
            self.is_processing = False
            self.process_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.progress_var.set(0)
            self.status_label.config(text="완료")
    
    def start_processing(self):
        """처리 시작"""
        if not self.s3_client or not self.db_conn:
            messagebox.showerror("오류", "먼저 연결 테스트를 수행해주세요.")
            return
        
        if not self.normal_periods and not self.anomaly_periods:
            messagebox.showerror("오류", "최소 하나의 기간을 설정해주세요.")
            return
        
        self.is_processing = True
        self.process_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # 별도 스레드에서 실행
        thread = threading.Thread(target=self.process_s3_files)
        thread.daemon = True
        thread.start()
    
    def stop_processing(self):
        """처리 중지"""
        self.is_processing = False
        self.log("처리 중지 요청...")


def main():
    # Docker 실행 확인
    print("TimescaleDB Docker 컨테이너를 시작하려면 다음 명령을 실행하세요:")
    print("docker-compose up -d")
    print("\n프로그램 시작 중...")
    
    root = tk.Tk()
    app = S3ToTimescaleDBApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()