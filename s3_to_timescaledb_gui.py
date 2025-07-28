# s3_to_timescaledb_gui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime, timedelta
import threading
import boto3
import psycopg2
from psycopg2 import pool
import numpy as np
from io import BytesIO
import os
from dotenv import load_dotenv
import time
import struct

from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import pandas as pd
from psycopg2.extras import execute_values
from collections import deque
import logging

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
        self.db_pool = None
        self.is_processing = False
        
        # 배치 처리 설정
        self.batch_size = 5000000  # 한 번에 삽입할 레코드 수
        self.file_batch_size = 1000  # 한 번에 처리할 파일 수
        self.commit_interval = 500  # 커밋 간격 (파일 수)
        self.max_workers = 20  # 동시 처리 워커 수
        self.verbose_logging = False  # 상세 로그 on/off
        
        # 통계 정보
        self.stats = {'processed_files': 0, 'total_records': 0, 'start_time': None}
        
        # 로깅 시스템 초기화
        self.setup_logging()
        
        # 로그 큐 및 버퍼 설정
        self.log_queue = queue.Queue()
        self.log_buffer = deque(maxlen=1000)  # 최대 1000줄만 유지
        self.last_log_update = time.time()
        
        # UI 생성
        self.create_ui()
        
        # 로그 업데이트 타이머 시작
        self.update_log_display()
        
        # 환경 변수 로드
        load_dotenv()
        
    def setup_logging(self):
        """파일 및 콘솔 로깅 설정"""
        # 로그 디렉토리 생성
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 로그 파일명 (타임스탬프 포함)
        log_filename = os.path.join(log_dir, f"s3_to_timescale_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 로거 설정
        self.logger = logging.getLogger("S3ToTimescaleDB")
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_filename = log_filename
        
    def create_ui(self):
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
        self.batch_size_var = tk.IntVar(value=5000000)
        batch_spinbox = ttk.Spinbox(machine_frame, from_=10000, to=2000000, increment=50000, 
                                   textvariable=self.batch_size_var, width=12)
        batch_spinbox.grid(row=1, column=1, padx=5)
        
        ttk.Label(machine_frame, text="동시 처리:").grid(row=1, column=2, sticky=tk.W, padx=20)
        self.workers_var = tk.IntVar(value=20)
        workers_spinbox = ttk.Spinbox(machine_frame, from_=1, to=30, increment=1,
                                     textvariable=self.workers_var, width=10)
        workers_spinbox.grid(row=1, column=3, padx=5)
        
        ttk.Label(machine_frame, text="파일 배치:").grid(row=1, column=4, sticky=tk.W, padx=5)
        self.file_batch_var = tk.IntVar(value=1000)
        file_batch_spinbox = ttk.Spinbox(machine_frame, from_=10, to=1000, increment=50,
                                        textvariable=self.file_batch_var, width=10)
        file_batch_spinbox.grid(row=1, column=5, padx=5)
        
        # 로그 상세도 체크박스 추가
        self.verbose_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(machine_frame, text="상세 로그", variable=self.verbose_log_var,
                       command=self.toggle_verbose_logging).grid(row=1, column=6, padx=5)
        
    def create_period_section(self, parent):
        # 기간 설정 프레임
        period_frame = ttk.LabelFrame(parent, text="날짜 범위 설정", padding="5")
        period_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 날짜 범위 입력
        date_frame = ttk.Frame(period_frame)
        date_frame.grid(row=0, column=0, padx=10, sticky=(tk.W, tk.E))
        
        ttk.Label(date_frame, text="시작일:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.start_date = ttk.Entry(date_frame, width=12)
        self.start_date.grid(row=0, column=1, padx=5)
        self.start_date.insert(0, (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        
        ttk.Label(date_frame, text="종료일:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.end_date = ttk.Entry(date_frame, width=12)
        self.end_date.grid(row=0, column=3, padx=5)
        self.end_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        # 선택된 날짜 범위 표시
        self.date_range_label = ttk.Label(date_frame, text="", font=('Arial', 9, 'italic'))
        self.date_range_label.grid(row=1, column=0, columnspan=4, pady=10)
        
    def create_action_section(self, parent):
        # 실행 버튼 프레임
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.process_btn = ttk.Button(action_frame, text="데이터 가져오기", command=self.start_processing, 
                                     style='Accent.TButton', padding=(20, 10))
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(action_frame, text="중지", command=self.stop_processing, 
                                  state='disabled', padding=(20, 10))
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
    def create_progress_section(self, parent):
        # 진행 상황 프레임
        progress_frame = ttk.LabelFrame(parent, text="진행 상황", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 로그 컨트롤 프레임
        log_control_frame = ttk.Frame(progress_frame)
        log_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(log_control_frame, text=f"로그 파일: {getattr(self, 'log_filename', 'N/A')}").pack(side=tk.LEFT, padx=5)
        ttk.Button(log_control_frame, text="로그 지우기", command=self.clear_log_display).pack(side=tk.RIGHT, padx=5)
        ttk.Button(log_control_frame, text="로그 복사", command=self.copy_log).pack(side=tk.RIGHT, padx=5)
        
        # 자동 스크롤 체크박스
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_control_frame, text="자동 스크롤", variable=self.auto_scroll_var).pack(side=tk.RIGHT, padx=5)
        
        # 진행률 바
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 상태 레이블
        self.status_label = ttk.Label(progress_frame, text="대기 중...")
        self.status_label.grid(row=2, column=0, sticky=tk.W, padx=5)
        
        # 통계 정보 레이블 추가
        self.stats_label = ttk.Label(progress_frame, text="")
        self.stats_label.grid(row=3, column=0, sticky=tk.W, padx=5)
        
        # 예상 시간 레이블
        self.eta_label = ttk.Label(progress_frame, text="")
        self.eta_label.grid(row=4, column=0, sticky=tk.W, padx=5)
        
        # 로그 텍스트
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=10, width=80, wrap=tk.WORD)
        self.log_text.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # 텍스트 태그 설정
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("SUCCESS", foreground="green")
        
        progress_frame.rowconfigure(5, weight=1)
        
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
            # 연결 풀 생성
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                1, self.max_workers + 1,  # min, max connections
                host="localhost",
                port="5432",
                database="pdm_db",
                user="pdm_user",
                password="pdm_password"
            )
            self.log("TimescaleDB 연결 성공!")
            
            # 테스트 연결
            test_conn = self.db_pool.getconn()
            
            # TimescaleDB extension 확인
            cur = test_conn.cursor()
            cur.execute("SELECT default_version FROM pg_available_extensions WHERE name = 'timescaledb';")
            result = cur.fetchone()
            if result:
                self.log(f"TimescaleDB 버전: {result[0]}")
            cur.close()
            self.db_pool.putconn(test_conn)
            
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
    
    def toggle_verbose_logging(self):
        """상세 로그 토글"""
        self.verbose_logging = self.verbose_log_var.get()
    
    def log(self, message, level="INFO"):
        """로그 메시지 추가 (큐에 추가)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # 파일 및 콘솔 로깅
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # GUI 로그 큐에 추가
        try:
            self.log_queue.put_nowait((formatted_message, level))
        except queue.Full:
            pass  # 큐가 가득 찬 경우 무시
    
    def update_log_display(self):
        """로그 디스플레이 업데이트 (100ms마다)"""
        try:
            # 큐에서 메시지 가져오기 (최대 50개씩)
            messages_to_add = []
            for _ in range(50):
                try:
                    message, level = self.log_queue.get_nowait()
                    messages_to_add.append((message, level))
                    self.log_buffer.append(message)  # 버퍼에도 추가
                except queue.Empty:
                    break
            
            # 메시지가 있으면 텍스트 위젯에 추가
            if messages_to_add:
                # 현재 스크롤 위치 저장
                current_pos = self.log_text.yview()
                
                for message, level in messages_to_add:
                    self.log_text.insert(tk.END, message + "\n", level)
                
                # 텍스트가 너무 길면 오래된 내용 삭제
                line_count = int(self.log_text.index('end-1c').split('.')[0])
                if line_count > 1000:
                    self.log_text.delete('1.0', f'{line_count-800}.0')
                
                # 자동 스크롤이 켜져 있으면 맨 아래로
                if self.auto_scroll_var.get():
                    self.log_text.see(tk.END)
                else:
                    # 원래 위치 복원
                    self.log_text.yview_moveto(current_pos[0])
        
        except Exception as e:
            print(f"로그 업데이트 오류: {e}")
        
        # 100ms 후에 다시 실행
        self.root.after(100, self.update_log_display)
    
    def clear_log_display(self):
        """로그 디스플레이 지우기"""
        self.log_text.delete('1.0', tk.END)
        self.log_buffer.clear()
        # 큐 비우기
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break
    
    def copy_log(self):
        """로그 복사"""
        try:
            # 현재 표시된 로그 텍스트 가져오기
            log_content = self.log_text.get('1.0', tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            messagebox.showinfo("복사 완료", "로그가 클립보드에 복사되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"로그 복사 중 오류 발생: {str(e)}")
    
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
    
    def create_tables(self):
        """TimescaleDB 테이블 생성"""
        conn = self.db_pool.getconn()
        try:
            cur = conn.cursor()
            
            # TimescaleDB extension 활성화
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
            # ACC 테이블
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
        
            # MIC 테이블
            cur.execute("""
                CREATE TABLE IF NOT EXISTS normal_mic_data (
                    time TIMESTAMPTZ NOT NULL,
                    machine_id TEXT NOT NULL,
                    mic_value INTEGER,
                    filename TEXT
                );
            """)
        
            # 하이퍼테이블로 변환
            tables = ['normal_acc_data', 'normal_mic_data']
            for table in tables:
                try:
                    # 청크 크기를 30일로 설정
                    cur.execute(f"SELECT create_hypertable('{table}', 'time', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);")
                    self.log(f"{table} 하이퍼테이블 생성/확인 완료")
                
                    # 압축 설정
                    cur.execute(f"""
                        ALTER TABLE {table} SET (
                            timescaledb.compress,
                            timescaledb.compress_segmentby = 'machine_id',
                            timescaledb.compress_orderby = 'time DESC'
                        );
                    """)
                
                    # 성능 설정
                    cur.execute(f"ALTER TABLE {table} SET (autovacuum_enabled = false);")
                    cur.execute(f"ALTER TABLE {table} SET (toast.autovacuum_enabled = false);")
                    
                except Exception as e:
                    self.log(f"{table} 하이퍼테이블 설정 중 오류 (이미 존재할 수 있음): {str(e)}")
        
            conn.commit()
            cur.close()
        finally:
            self.db_pool.putconn(conn)
    
    def load_acc_dat_from_s3(self, bucket, key):
        """S3에서 ACC DAT 파일 읽기"""
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        data_stream = BytesIO(obj['Body'].read())
        
        # 파일 크기 확인
        file_size = data_stream.seek(0, 2)
        data_stream.seek(0)
        
        # 각 세트는 3000개 int16 (6000바이트) + 8바이트 스킵 = 6008바이트
        sets = file_size // 6008
        total_samples = sets * 1000
        
        # 전체 데이터를 먼저 읽기
        all_data = np.empty((total_samples, 3), dtype=np.float64)
        read_count = 0
        
        for i in range(sets):
            # 3000개 int16 읽기
            chunk_bytes = data_stream.read(6000)
            if len(chunk_bytes) < 6000:
                break
                
            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
            chunk_data = chunk.reshape(-1, 3)
            
            csize = min(1000, total_samples - read_count)
            all_data[read_count:read_count+csize, :] = chunk_data[:csize, :].astype(np.float64) * 0.000488
            read_count += csize
            
            # 8바이트 스킵
            data_stream.read(8)
        
        # 샘플링: 각 초마다 앞 30개씩 추출 (5초간)
        sampling_rate = 1666  # ACC 샘플링 레이트
        samples_per_second = 30
        sampled_data = []
        
        for second in range(5):  # 0~4초 (총 5초)
            start_idx = second * sampling_rate
            if start_idx + samples_per_second <= read_count:
                sampled_data.append(all_data[start_idx:start_idx + samples_per_second])
            else:
                break
        
        if sampled_data:
            return np.vstack(sampled_data)
        else:
            return np.empty((0, 3), dtype=np.float64)
    
    def load_mic_dat_from_s3(self, bucket, key):
        """S3에서 MIC DAT 파일 읽기"""
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        data_stream = BytesIO(obj['Body'].read())
        
        # 파일 크기 확인
        file_size = data_stream.seek(0, 2)
        data_stream.seek(0)
        
        # 각 세트는 1000개 int16 (2000바이트) + 8바이트 스킵 = 2008바이트
        sets = file_size // 2008
        total_samples = sets * 1000
        
        # 전체 데이터를 먼저 읽기
        all_data = np.empty(total_samples, dtype=np.int16)
        
        for i in range(sets):
            # 1000개 int16 읽기
            chunk_bytes = data_stream.read(2000)
            if len(chunk_bytes) < 2000:
                break
                
            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
            all_data[i*1000:(i+1)*1000] = chunk
            
            # 8바이트 스킵
            data_stream.read(8)
        
        # 샘플링: 각 초마다 앞 30개씩 추출 (5초간)
        sampling_rate = 8000  # MIC 샘플링 레이트
        samples_per_second = 30
        sampled_data = []
        
        for second in range(5):  # 0~4초 (총 5초)
            start_idx = second * sampling_rate
            if start_idx + samples_per_second <= i*1000+len(chunk):
                sampled_data.append(all_data[start_idx:start_idx + samples_per_second])
            else:
                break
        
        if sampled_data:
            return np.hstack(sampled_data)
        else:
            return np.empty(0, dtype=np.int16)
    
    def insert_buffer_data(self, table_name, buffer, conn):
        """버퍼 데이터를 COPY로 삽입"""
        cur = conn.cursor()
        
        try:
            buffer.seek(0)
            
            # 테이블에 따라 컬럼 결정
            if 'acc' in table_name:
                columns = "(time, machine_id, x, y, z, filename)"
            else:  # mic
                columns = "(time, machine_id, mic_value, filename)"
            
            # COPY 명령
            cur.copy_expert(
                f"COPY {table_name} {columns} FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t')",
                buffer
            )
            
            # 삽입된 행 수 계산
            buffer.seek(0)
            inserted = sum(1 for _ in buffer)
            
            conn.commit()
            
            self.log(f"  ✅ {table_name}에 {inserted:,}개 레코드 삽입 완료", "SUCCESS")
            self.stats['total_records'] += inserted
                
            return inserted
            
        except Exception as e:
            self.log(f"  ❌ {table_name} 삽입 오류: {str(e)}", "ERROR")
            import traceback
            self.log(f"  상세: {traceback.format_exc()}")
            conn.rollback()
            raise
        finally:
            cur.close()
    
    def process_file_batch(self, file_batch, machine_id):
        """파일 배치를 처리하고 데이터를 준비 - 벡터화 버전"""
        from io import StringIO
        
        # StringIO 버퍼 사용
        acc_buffer = StringIO()
        mic_buffer = StringIO()
        
        bucket = self.bucket_var.get()
        processed_files = 0
        skipped_files = 0
        
        for sensor, key in file_batch:
            if not self.is_processing:
                break
                
            filename = os.path.basename(key)
            file_date = self.parse_filename_date(filename)
            
            if not file_date:
                self.log(f"  ⚠️ 날짜 파싱 실패: {filename}", "WARNING")
                skipped_files += 1
                continue
            
            try:
                if self.verbose_logging:
                    self.log(f"  처리 중: {filename} (날짜: {file_date})")
                
                if sensor == 'acc':
                    data = self.load_acc_dat_from_s3(bucket, key)
                    n_samples = len(data)
                    if self.verbose_logging:
                        self.log(f"    - ACC 데이터 로드: {n_samples} 샘플")
                    
                    # 샘플링된 데이터의 타임스탬프 계산
                    # 각 초마다 30개씩, 총 5초
                    timestamps = []
                    for second in range(5):
                        base_time = file_date + pd.Timedelta(seconds=second)
                        # 각 초의 처음 30개 샘플에 대한 타임스탬프
                        for sample_idx in range(30):
                            # ACC는 1666Hz이므로 샘플 간격은 약 600.6μs
                            sample_time = base_time + pd.Timedelta(microseconds=int(sample_idx * 600.6))
                            timestamps.append(sample_time)
                    
                    timestamps = timestamps[:n_samples]  # 실제 샘플 수에 맞춤
                    
                    # 벡터화된 쓰기
                    for i in range(0, n_samples, 10000):  # 10000개씩 처리
                        end_idx = min(i + 10000, n_samples)
                        for j in range(i, end_idx):
                            acc_buffer.write(f"{timestamps[j].isoformat()}\t{machine_id}\t"
                                       f"{data[j,0]:.6f}\t{data[j,1]:.6f}\t{data[j,2]:.6f}\t{filename}\n")
                    
                    if self.verbose_logging:
                        self.log(f"    - ACC 버퍼에 {n_samples} 레코드 추가")
                    processed_files += 1
                        
                else:  # mic
                    data = self.load_mic_dat_from_s3(bucket, key)
                    n_samples = len(data)
                    if self.verbose_logging:
                        self.log(f"    - MIC 데이터 로드: {n_samples} 샘플")
                    
                    # 샘플링된 데이터의 타임스탬프 계산
                    timestamps = []
                    for second in range(5):
                        base_time = file_date + pd.Timedelta(seconds=second)
                        # 각 초의 처음 30개 샘플에 대한 타임스탬프
                        for sample_idx in range(30):
                            # MIC는 8000Hz이므로 샘플 간격은 125μs
                            sample_time = base_time + pd.Timedelta(microseconds=int(sample_idx * 125))
                            timestamps.append(sample_time)
                    
                    timestamps = timestamps[:n_samples]  # 실제 샘플 수에 맞춤
                    
                    # 벡터화된 쓰기
                    for i in range(0, n_samples, 10000):  # 10000개씩 처리
                        end_idx = min(i + 10000, n_samples)
                        for j in range(i, end_idx):
                            mic_buffer.write(f"{timestamps[j].isoformat()}\t{machine_id}\t{data[j]}\t{filename}\n")
                    
                    if self.verbose_logging:
                        self.log(f"    - MIC 버퍼에 {n_samples} 레코드 추가")
                    processed_files += 1
                        
            except Exception as e:
                self.log(f"  ❌ 파일 처리 오류 ({filename}): {str(e)}", "ERROR")
                import traceback
                self.log(f"상세 오류: {traceback.format_exc()}")
                skipped_files += 1
                continue
        
        self.log(f"\n배치 처리 결과:")
        self.log(f"  - 처리 완료: {processed_files}개")
        self.log(f"  - 건너뛴 파일: {skipped_files}개")
        self.log(f"  - ACC 버퍼 크기: {acc_buffer.tell()} bytes")
        self.log(f"  - MIC 버퍼 크기: {mic_buffer.tell()} bytes")
        
        # 버퍼 크기를 저장 (seek 전에)
        acc_buffer_size = acc_buffer.tell()
        mic_buffer_size = mic_buffer.tell()
        
        # 버퍼 위치를 처음으로 리셋 (중요!)
        acc_buffer.seek(0)
        mic_buffer.seek(0)
        
        self.log(f"버퍼 위치 리셋 완료")
        
        return {
            'acc': acc_buffer,
            'mic': mic_buffer,
            'acc_size': acc_buffer_size,
            'mic_size': mic_buffer_size
        }
    
    def compress_table_chunks(self, table_name):
        """테이블의 압축되지 않은 청크 압축"""
        conn = self.db_pool.getconn()
        try:
            cur = conn.cursor()
        
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
                self.log(f"{table_name} 압축 중 오류: {str(e)}", "WARNING")
        
            conn.commit()
            cur.close()
        finally:
            self.db_pool.putconn(conn)
    
    def process_s3_files(self):
        """S3 파일 처리 메인 로직"""
        try:
            # 시스템 메모리 확인
            import psutil
            total_memory = psutil.virtual_memory().total / (1024**3)  # GB
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            self.log(f"시스템 메모리: 총 {total_memory:.1f}GB, 사용 가능 {available_memory:.1f}GB")
            
            # PostgreSQL 세션 레벨 성능 설정 - 메모리의 80% 활용
            def apply_session_settings(conn):
                cur = conn.cursor()
                work_mem = int(available_memory * 0.2 * 1024)  # 20% of available memory in MB
                maintenance_mem = int(available_memory * 0.3 * 1024)  # 30% of available memory in MB
                
                session_settings = [
                    "SET synchronous_commit = OFF;",
                    f"SET work_mem = '{work_mem}MB';",
                    f"SET maintenance_work_mem = '{maintenance_mem}MB';",
                    f"SET temp_buffers = '{int(available_memory * 0.1 * 1024)}MB';",
                    "SET effective_io_concurrency = 200;",
                    "SET max_parallel_workers_per_gather = 8;",
                ]
                
                self.log(f"PostgreSQL 설정: work_mem={work_mem}MB, maintenance_work_mem={maintenance_mem}MB")
                
                for setting in session_settings:
                    try:
                        cur.execute(setting)
                    except Exception as e:
                        self.log(f"설정 적용 건너뜀: {setting} - {str(e)}", "WARNING")
                
                conn.commit()
                cur.close()
            
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
            
            # 날짜 범위 가져오기
            start_date = datetime.strptime(self.start_date.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date.get(), "%Y-%m-%d")
            # 시간 포함한 비교를 위해 end_date는 23:59:59로 설정
            end_date = end_date.replace(hour=23, minute=59, second=59)
            
            self.log(f"날짜 범위: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
            
            # 센서별로 전체 파일 목록 가져오기 (한 번의 API 호출)
            total_files = 0
            for sensor in sensors:
                sensor_files = 0
                prefix = f"{machine_id}/raw_dat/{sensor}/"
                self.log(f"{sensor.upper()} 파일 목록 가져오는 중 (전체 스캔)...")
                
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith('.dat'):
                                # 파일명에서 날짜 추출하여 범위 확인
                                filename = os.path.basename(obj['Key'])
                                file_date = self.parse_filename_date(filename)
                                
                                if file_date and start_date <= file_date <= end_date:
                                    all_files.append((sensor, obj['Key']))
                                    sensor_files += 1
                
                self.log(f"{sensor.upper()}: {sensor_files}개 파일")
                total_files += sensor_files
            
            # 파일 이름으로 정렬 (센서별, 날짜순)
            all_files.sort(key=lambda x: (x[0], x[1]))
            
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
                
                self.log(f"\n파일 배치 처리 시작: 총 {len(file_batches)}개 배치")
                
                for batch_idx, file_batch in enumerate(file_batches):
                    if not self.is_processing:
                        break
                    
                    self.log(f"\n배치 {batch_idx+1}/{len(file_batches)} 제출 ({len(file_batch)}개 파일)")
                    future = executor.submit(self.process_file_batch, file_batch, machine_id)
                    futures.append((batch_idx, future))
                    
                    # 디버깅: 제출된 future 확인
                    self.log(f"Future 제출됨: batch_idx={batch_idx}, future={future}")
                    
                # 결과 수집 및 DB 삽입
                self.log(f"\n배치 결과 수집 및 DB 삽입 시작")
                for batch_idx, future in futures:
                    if not self.is_processing:
                        break
                        
                    try:
                        self.log(f"배치 {batch_idx+1} 결과 대기 중...")
                        result = future.result(timeout=300)  # 5분 타임아웃
                        self.log(f"\n배치 {batch_idx+1} 결과 수신")
                        
                        # 디버깅: 결과 확인
                        self.log(f"ACC 버퍼 크기: {result.get('acc_size', 0)} bytes")
                        self.log(f"MIC 버퍼 크기: {result.get('mic_size', 0)} bytes")
                        
                        # 각 워커에 대해 별도의 DB 연결 사용
                        conn = self.db_pool.getconn()
                        try:
                            # 각 연결에 세션 설정 적용
                            apply_session_settings(conn)
                            
                            # 배치 데이터 삽입
                            if result.get('acc_size', 0) > 0:
                                self.log(f"  ACC 데이터 삽입 중... (버퍼 크기: {result['acc_size']} bytes)")
                                inserted = self.insert_buffer_data('normal_acc_data', result['acc'], conn)
                                self.log(f"  ACC 삽입 완료: {inserted}개 레코드")
                            else:
                                self.log(f"  ACC 버퍼가 비어있음")
                            
                            if result.get('mic_size', 0) > 0:
                                self.log(f"  MIC 데이터 삽입 중... (버퍼 크기: {result['mic_size']} bytes)")
                                inserted = self.insert_buffer_data('normal_mic_data', result['mic'], conn)
                                self.log(f"  MIC 삽입 완료: {inserted}개 레코드")
                            else:
                                self.log(f"  MIC 버퍼가 비어있음")
                                
                            # 디버깅: 커밋 확인
                            self.log(f"  DB 커밋 중...")
                            conn.commit()
                            self.log(f"  DB 커밋 완료")
                        finally:
                            self.db_pool.putconn(conn)
                            self.log(f"  DB 연결 반환 완료")
                        
                        # 통계 업데이트
                        self.stats['processed_files'] += self.file_batch_size
                        
                        # 상세 통계 로그
                        if batch_idx % 10 == 0:
                            self.log(f"배치 {batch_idx+1}/{len(file_batches)} 완료:")
                            self.log(f"  - 총 파일: {self.stats['processed_files']:,}")
                            self.log(f"  - 총 레코드: {self.stats['total_records']:,}")
                            
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
                            
                            # UI 업데이트는 1초에 한 번만
                            current_time = time.time()
                            if current_time - self.last_log_update > 1.0:
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
                                self.last_log_update = current_time
                        
                        # 주기적 압축 (200개 배치마다)
                        if batch_idx % 200 == 0 and batch_idx > 0:
                            self.log("청크 압축 중...")
                            for table_name in ['normal_acc_data', 'normal_mic_data']:
                                self.compress_table_chunks(table_name)
                                
                    except Exception as e:
                        self.log(f"배치 {batch_idx+1} 처리 오류: {str(e)}", "ERROR")
                        import traceback
                        self.log(f"상세 오류:\n{traceback.format_exc()}", "ERROR")
                        continue
            
            # autovacuum 다시 활성화 및 인덱스 생성
            conn = self.db_pool.getconn()
            try:
                # VACUUM을 위해 별도의 autocommit 연결 사용
                conn.autocommit = True
                cur = conn.cursor()
                
                for table in ['normal_acc_data', 'normal_mic_data']:
                    try:
                        cur.execute(f"ALTER TABLE {table} SET (autovacuum_enabled = true);")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_machine_time ON {table} (machine_id, time DESC);")
                        self.log(f"{table} 인덱스 생성 완료")
                        
                        cur.execute(f"VACUUM ANALYZE {table};")
                        self.log(f"{table} VACUUM ANALYZE 완료")
                    except Exception as e:
                        self.log(f"{table} 후처리 중 오류: {str(e)}", "WARNING")
                
                cur.close()
            finally:
                self.db_pool.putconn(conn)
            
            # 최종 압축
            self.log("\n최종 압축 중...")
            for table_name in ['normal_acc_data', 'normal_mic_data']:
                self.compress_table_chunks(table_name)
            
            # 처리 시간 계산
            total_time = time.time() - self.stats['start_time']
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            
            self.log(f"\n처리 완료!", "SUCCESS")
            self.log(f"총 처리 시간: {hours}시간 {minutes}분")
            self.log(f"총 파일: {self.stats['processed_files']:,}")
            self.log(f"총 레코드: {self.stats['total_records']:,}")
            self.log(f"로그 파일: {self.log_filename}")
            
            messagebox.showinfo("완료", 
                f"처리 완료!\n\n"
                f"처리 시간: {hours}시간 {minutes}분\n"
                f"총 파일: {self.stats['processed_files']:,}\n"
                f"총 레코드: {self.stats['total_records']:,}")
            
        except Exception as e:
            self.log(f"처리 중 오류 발생: {str(e)}", "ERROR")
            messagebox.showerror("오류", f"처리 중 오류 발생: {str(e)}")
        
        finally:
            self.is_processing = False
            self.process_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.progress_var.set(0)
            self.status_label.config(text="완료")
    
    def start_processing(self):
        """처리 시작"""
        if not self.s3_client or not self.db_pool:
            messagebox.showerror("오류", "먼저 연결 테스트를 수행해주세요.")
            return
        
        try:
            start_date = datetime.strptime(self.start_date.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date.get(), "%Y-%m-%d")
            if start_date > end_date:
                messagebox.showerror("오류", "시작일이 종료일보다 늦습니다.")
                return
        except ValueError:
            messagebox.showerror("오류", "날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)")
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
        self.stop_btn.config(state='disabled')
        self.process_btn.config(state='normal')


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