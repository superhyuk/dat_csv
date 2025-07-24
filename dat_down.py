import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

class S3DataDownloader:
    def __init__(self, root):
        self.root = root
        self.root.title("S3 데이터 다운로더")
        self.root.geometry("700x850")
        
        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        
        # 환경 변수 초기화
        self.aws_access_key = None
        self.aws_secret_key = None
        self.aws_region = None
        self.bucket_name = None
        
        # 자동 저장 경로 설정 (실행 디렉토리의 하위폴더)
        self.base_save_path = os.path.join(os.getcwd(), "S3_Downloads")
        
        # S3 클라이언트
        self.s3_client = None
        
        # 다운로드 설정
        self.max_workers = 10  # 동시 다운로드 스레드 수
        self.download_queue = queue.Queue()
        
        # UI 생성
        self.create_ui()
        
    def create_ui(self):
        """UI 생성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 타이틀
        title_label = ttk.Label(main_frame, text="S3 데이터 다운로더", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # .env 파일 로드 섹션
        env_frame = ttk.LabelFrame(main_frame, text="환경 설정", padding="10")
        env_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.env_status = ttk.Label(env_frame, text=".env 파일이 로드되지 않았습니다.", foreground="red")
        self.env_status.grid(row=0, column=0, padx=5)
        
        ttk.Button(env_frame, text=".env 파일 선택", command=self.load_env_file).grid(row=0, column=1, padx=5)
        
        # 연결 테스트 버튼
        self.test_connection_button = ttk.Button(env_frame, text="연결 테스트", command=self.test_connection, state='disabled')
        self.test_connection_button.grid(row=0, column=2, padx=5)
        
        # 데이터 타입 선택 (acc/mic)
        type_frame = ttk.LabelFrame(main_frame, text="데이터 타입 선택", padding="10")
        type_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(type_frame, text="데이터 타입:", font=('Arial', 12)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_type_var = tk.StringVar(value="mic")
        
        data_type_frame = ttk.Frame(type_frame)
        data_type_frame.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Radiobutton(data_type_frame, text="MIC 데이터", variable=self.data_type_var, 
                       value="mic").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(data_type_frame, text="ACC 데이터", variable=self.data_type_var, 
                       value="acc").pack(side=tk.LEFT, padx=10)
        
        # 머신 ID 선택
        ttk.Label(type_frame, text="머신 ID:", font=('Arial', 12)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.machine_var = tk.StringVar()
        machine_combo = ttk.Combobox(type_frame, textvariable=self.machine_var, width=30)
        machine_combo['values'] = ('HOTCHAMBER_M2', 'CURINGOVEN_M1')
        machine_combo.grid(row=1, column=1, pady=5, sticky=(tk.W, tk.E))
        machine_combo.state(['readonly'])
        machine_combo.set('HOTCHAMBER_M2')  # 기본값 설정
        
        # 버킷 및 경로 선택
        path_frame = ttk.LabelFrame(main_frame, text="S3 버킷 설정", padding="10")
        path_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # 버킷 선택
        ttk.Label(path_frame, text="버킷 이름:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.bucket_var = tk.StringVar()
        self.bucket_combo = ttk.Combobox(path_frame, textvariable=self.bucket_var, width=40)
        self.bucket_combo.grid(row=0, column=1, pady=5, sticky=(tk.W, tk.E))
        self.bucket_combo.bind('<<ComboboxSelected>>', self.on_bucket_selected)
        
        ttk.Button(path_frame, text="버킷 목록 불러오기", command=self.load_buckets).grid(row=0, column=2, padx=5)
        
        # 날짜 선택 프레임
        date_frame = ttk.LabelFrame(main_frame, text="날짜 범위 선택", padding="10")
        date_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # 하루치 다운로드 옵션
        self.single_day_var = tk.BooleanVar()
        single_day_cb = ttk.Checkbutton(date_frame, text="하루치 데이터만 다운로드", 
                                        variable=self.single_day_var, command=self.toggle_single_day)
        single_day_cb.grid(row=0, column=0, columnspan=3, pady=5, sticky=tk.W)
        
        # 시작 날짜
        ttk.Label(date_frame, text="시작일:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.start_year = tk.StringVar(value=str(datetime.now().year))
        self.start_month = tk.StringVar(value=str(datetime.now().month).zfill(2))
        self.start_day = tk.StringVar(value=str(datetime.now().day).zfill(2))
        
        start_frame = ttk.Frame(date_frame)
        start_frame.grid(row=1, column=1, sticky=tk.W)
        
        self.start_year_entry = ttk.Entry(start_frame, textvariable=self.start_year, width=6)
        self.start_year_entry.pack(side=tk.LEFT)
        ttk.Label(start_frame, text="/").pack(side=tk.LEFT)
        self.start_month_entry = ttk.Entry(start_frame, textvariable=self.start_month, width=4)
        self.start_month_entry.pack(side=tk.LEFT)
        ttk.Label(start_frame, text="/").pack(side=tk.LEFT)
        self.start_day_entry = ttk.Entry(start_frame, textvariable=self.start_day, width=4)
        self.start_day_entry.pack(side=tk.LEFT)
        
        # 종료 날짜
        self.end_label = ttk.Label(date_frame, text="종료일:")
        self.end_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        self.end_year = tk.StringVar(value=str(datetime.now().year))
        self.end_month = tk.StringVar(value=str(datetime.now().month).zfill(2))
        self.end_day = tk.StringVar(value=str(datetime.now().day).zfill(2))
        
        self.end_frame = ttk.Frame(date_frame)
        self.end_frame.grid(row=2, column=1, sticky=tk.W)
        
        self.end_year_entry = ttk.Entry(self.end_frame, textvariable=self.end_year, width=6)
        self.end_year_entry.pack(side=tk.LEFT)
        ttk.Label(self.end_frame, text="/").pack(side=tk.LEFT)
        self.end_month_entry = ttk.Entry(self.end_frame, textvariable=self.end_month, width=4)
        self.end_month_entry.pack(side=tk.LEFT)
        ttk.Label(self.end_frame, text="/").pack(side=tk.LEFT)
        self.end_day_entry = ttk.Entry(self.end_frame, textvariable=self.end_day, width=4)
        self.end_day_entry.pack(side=tk.LEFT)
        
        # 저장 경로 섹션
        save_frame = ttk.LabelFrame(main_frame, text="저장 위치", padding="10")
        save_frame.grid(row=5, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.save_path_var = tk.StringVar(value=self.base_save_path)
        save_path_entry = ttk.Entry(save_frame, textvariable=self.save_path_var, width=50, state='readonly')
        save_path_entry.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Button(save_frame, text="폴더 선택", command=self.select_save_folder).grid(row=0, column=1, padx=5)
        
        # 저장 구조 설명
        ttk.Label(save_frame, text="저장 구조: /저장경로/머신ID/데이터타입/날짜/").grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Label(save_frame, text="S3 구조: s3://버킷명/머신ID/raw_dat/데이터타입/", 
                 font=('Arial', 9, 'italic')).grid(row=2, column=0, columnspan=2, pady=2)
        
        # 다운로드 설정 프레임
        settings_frame = ttk.LabelFrame(main_frame, text="다운로드 설정", padding="10")
        settings_frame.grid(row=6, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(settings_frame, text="동시 다운로드 수:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.workers_var = tk.IntVar(value=10)
        workers_spinbox = ttk.Spinbox(settings_frame, from_=1, to=50, textvariable=self.workers_var, width=10)
        workers_spinbox.grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(settings_frame, text="(1-50, 높을수록 빠르지만 네트워크 부하 증가)", 
                 font=('Arial', 9, 'italic')).grid(row=0, column=2, padx=10, sticky=tk.W)
        
        # 진행 상황
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=7, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.progress = ttk.Progressbar(progress_frame, length=600, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="대기 중...")
        self.status_label.pack(pady=5)
        
        # 로그 영역
        log_frame = ttk.LabelFrame(main_frame, text="진행 로그", padding="10")
        log_frame.grid(row=8, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = tk.Text(log_frame, height=6, width=70, state='disabled')
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=20)
        
        self.download_button = ttk.Button(button_frame, text="다운로드 시작", 
                                        command=self.start_download, width=20, padding="10")
        self.download_button.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="저장 폴더 열기", 
                  command=self.open_save_folder, width=20, padding="10").pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="종료", command=self.root.quit, width=20, padding="10").pack(side=tk.LEFT, padx=10)
        
        # 컬럼/행 설정
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 초기 로그 메시지
        self.log(f"저장 경로 설정: {self.base_save_path}")
    
    def toggle_single_day(self):
        """하루치 다운로드 선택 시 종료일 입력 비활성화"""
        if self.single_day_var.get():
            # 종료일 입력 비활성화
            self.end_label.grid_forget()
            self.end_frame.grid_forget()
            # 시작일로 종료일 복사
            self.end_year.set(self.start_year.get())
            self.end_month.set(self.start_month.get())
            self.end_day.set(self.start_day.get())
        else:
            # 종료일 입력 활성화
            self.end_label.grid(row=2, column=0, sticky=tk.W, pady=5)
            self.end_frame.grid(row=2, column=1, sticky=tk.W)
    
    def load_env_file(self):
        """환경 변수 파일 선택 및 로드"""
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
                
                # 로드된 값 확인
                self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
                self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
                self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
                self.bucket_name = os.getenv('S3_BUCKET_NAME')
                
                # 디버깅 정보 출력
                self.log("로드된 환경 변수:")
                self.log(f"  AWS_ACCESS_KEY_ID: {'있음' if self.aws_access_key else '없음'}")
                self.log(f"  AWS_SECRET_ACCESS_KEY: {'있음' if self.aws_secret_key else '없음'}")
                self.log(f"  AWS_REGION: {self.aws_region}")
                self.log(f"  S3_BUCKET_NAME: {self.bucket_name if self.bucket_name else '없음'}")
                
                if not all([self.aws_access_key, self.aws_secret_key]):
                    messagebox.showerror("오류", ".env 파일에 필요한 정보가 없습니다.\n필요한 항목: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
                    self.env_status.config(text=".env 파일이 유효하지 않습니다.", foreground="red")
                else:
                    self.env_status.config(text=f".env 파일 로드 완료", foreground="green")
                    self.test_connection_button.config(state='normal')
                    
                    # 버킷 이름이 있으면 자동으로 설정
                    if self.bucket_name:
                        self.bucket_var.set(self.bucket_name)
                    
                    messagebox.showinfo("성공", ".env 파일이 성공적으로 로드되었습니다.")
                    self.log(f".env 파일 로드 완료: {env_path}")
                
            except Exception as e:
                messagebox.showerror("오류", f".env 파일 로드 중 오류 발생: {str(e)}")
                self.env_status.config(text=".env 파일 로드 실패", foreground="red")
                self.log(f"오류: {str(e)}")
    
    def test_connection(self):
        """S3 연결 테스트"""
        self.log("S3 연결 테스트 시작...")
        self.test_connection_button.config(text="테스트 중...", state='disabled')
        
        def test_async():
            try:
                # S3 클라이언트 생성
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                
                # 버킷 목록 가져오기 (연결 테스트)
                response = self.s3_client.list_buckets()
                bucket_count = len(response['Buckets'])
                
                self.root.after(0, lambda: self.log(f"✅ S3 연결 성공! {bucket_count}개의 버킷 발견"))
                
                # 버킷 목록 업데이트
                bucket_names = [bucket['Name'] for bucket in response['Buckets']]
                self.root.after(0, lambda: self.bucket_combo.configure(values=bucket_names))
                
                self.root.after(0, lambda: messagebox.showinfo("연결 성공", 
                    f"S3 연결에 성공했습니다.\n\n발견된 버킷 수: {bucket_count}"))
                    
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.log(f"❌ 연결 실패: {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("연결 실패", 
                    f"S3 연결에 실패했습니다.\n\n오류: {error_msg}"))
            finally:
                self.root.after(0, lambda: self.test_connection_button.config(text="연결 테스트", state='normal'))
        
        thread = threading.Thread(target=test_async)
        thread.start()
    
    def load_buckets(self):
        """S3 버킷 목록 불러오기"""
        if not self.s3_client:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
            except Exception as e:
                messagebox.showerror("오류", f"S3 클라이언트 생성 실패: {str(e)}")
                return
        
        try:
            response = self.s3_client.list_buckets()
            bucket_names = [bucket['Name'] for bucket in response['Buckets']]
            self.bucket_combo['values'] = bucket_names
            self.log(f"{len(bucket_names)}개의 버킷을 불러왔습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"버킷 목록 불러오기 실패: {str(e)}")
            self.log(f"버킷 목록 불러오기 실패: {str(e)}")
    
    def on_bucket_selected(self, event):
        """버킷 선택 시 호출"""
        self.log(f"선택된 버킷: {self.bucket_var.get()}")
    
    def select_save_folder(self):
        """저장 폴더 선택"""
        folder_path = filedialog.askdirectory(title="저장 위치 선택", initialdir=self.save_path_var.get())
        if folder_path:
            self.save_path_var.set(folder_path)
            self.log(f"저장 경로 변경: {folder_path}")
    
    def open_save_folder(self):
        """저장 폴더 열기"""
        try:
            save_path = self.save_path_var.get()
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            
            if os.name == 'nt':  # Windows
                os.startfile(save_path)
            elif os.name == 'posix':  # macOS and Linux
                subprocess.run(['open', save_path])
        except Exception as e:
            messagebox.showinfo("알림", f"저장 경로: {self.save_path_var.get()}")
    
    def validate_inputs(self):
        """입력값 검증"""
        if not all([self.aws_access_key, self.aws_secret_key]):
            messagebox.showerror("오류", ".env 파일을 먼저 로드해주세요.")
            return False
            
        if not self.bucket_var.get():
            messagebox.showerror("오류", "버킷을 선택해주세요.")
            return False
            
        if not self.machine_var.get():
            messagebox.showerror("오류", "머신 ID를 선택해주세요.")
            return False
            
        try:
            start_date = datetime(int(self.start_year.get()), 
                                int(self.start_month.get()), 
                                int(self.start_day.get()))
            
            if self.single_day_var.get():
                # 하루치 다운로드인 경우 종료일을 시작일과 동일하게 설정
                end_date = start_date
            else:
                end_date = datetime(int(self.end_year.get()), 
                                  int(self.end_month.get()), 
                                  int(self.end_day.get()))
                              
            if end_date < start_date:
                messagebox.showerror("오류", "종료일이 시작일보다 이전입니다.")
                return False
                
            return True
            
        except ValueError as e:
            messagebox.showerror("오류", f"유효한 날짜를 입력해주세요: {str(e)}")
            return False
    
    def download_single_file(self, file_info):
        """단일 파일 다운로드 (멀티스레딩용)"""
        bucket = file_info['bucket']
        key = file_info['key']
        local_path = file_info['local_path']
        
        try:
            self.s3_client.download_file(bucket, key, local_path)
            return {'success': True, 'key': key}
        except Exception as e:
            return {'success': False, 'key': key, 'error': str(e)}
    
    def download_files_parallel(self, dat_files, date_folder, current_date):
        """병렬로 파일 다운로드"""
        # 다운로드 작업 준비
        download_tasks = []
        for file_obj in dat_files:
            file_key = file_obj['Key']
            file_name = os.path.basename(file_key)
            local_path = os.path.join(date_folder, file_name)
            
            download_tasks.append({
                'bucket': self.bucket_var.get(),
                'key': file_key,
                'local_path': local_path
            })
        
        # 멀티스레딩으로 다운로드
        downloaded_count = 0
        failed_count = 0
        start_time = datetime.now()
        
        with ThreadPoolExecutor(max_workers=self.workers_var.get()) as executor:
            # 모든 다운로드 작업 제출
            future_to_task = {executor.submit(self.download_single_file, task): task 
                             for task in download_tasks}
            
            # 완료된 작업 처리
            for i, future in enumerate(as_completed(future_to_task)):
                result = future.result()
                
                if result['success']:
                    downloaded_count += 1
                else:
                    failed_count += 1
                    self.log(f"  다운로드 실패: {os.path.basename(result['key'])} - {result.get('error', 'Unknown error')}")
                
                # 진행률 업데이트
                if (i + 1) % 50 == 0 or (i + 1) == len(download_tasks):
                    progress = (i + 1) / len(download_tasks) * 100
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    speed = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                    
                    self.root.after(0, lambda p=progress, s=speed, d=current_date, i=i+1, t=len(download_tasks): 
                        self.status_label.config(text=f"{d.strftime('%Y-%m-%d')} 다운로드 중... ({i}/{t}) - {s:.1f} 파일/초"))
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        avg_speed = downloaded_count / elapsed_time if elapsed_time > 0 else 0
        
        return downloaded_count, failed_count, elapsed_time, avg_speed
    
    def start_download(self):
        """다운로드 시작"""
        if not self.validate_inputs():
            return
            
        self.download_button.config(state='disabled')
        self.progress['value'] = 0
        self.status_label.config(text="다운로드 준비 중...")
        
        # 동시 다운로드 수 업데이트
        self.max_workers = self.workers_var.get()
        
        # 별도 스레드에서 다운로드 실행
        thread = threading.Thread(target=self.download_data)
        thread.start()
    
    def download_data(self):
        """데이터 다운로드 실행"""
        try:
            # S3 클라이언트 생성
            if not self.s3_client:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
            
            # 날짜 범위 설정
            start_date = datetime(int(self.start_year.get()), 
                                int(self.start_month.get()), 
                                int(self.start_day.get()))
            
            if self.single_day_var.get():
                end_date = start_date
                self.log(f"하루치 다운로드: {start_date.strftime('%Y-%m-%d')}")
            else:
                end_date = datetime(int(self.end_year.get()), 
                                  int(self.end_month.get()), 
                                  int(self.end_day.get()))
                self.log(f"기간별 다운로드: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
            
            current_date = start_date
            total_days = (end_date - start_date).days + 1
            processed_days = 0
            total_files = 0
            
            # 데이터 타입과 머신 ID 가져오기
            data_type = self.data_type_var.get()
            machine_id = self.machine_var.get()
            
            while current_date <= end_date:
                # S3 경로 생성: {machine_id}/raw_dat/{data_type}/{YYYYMMDD}_로 시작하는 모든 파일
                date_str = current_date.strftime('%Y%m%d')
                prefix = f"{machine_id}/raw_dat/{data_type}/{date_str}_"
                
                # 상태 업데이트
                self.root.after(0, lambda d=current_date: self.status_label.config(
                    text=f"{d.strftime('%Y-%m-%d')} 데이터 다운로드 중..."))
                
                # 해당 날짜의 파일 목록 가져오기
                try:
                    self.log(f"\n{current_date.strftime('%Y-%m-%d')} 파일 검색 중...")
                    self.log(f"  경로: {machine_id}/raw_dat/{data_type}/")
                    self.log(f"  prefix: {prefix}")
                    
                    # 페이지네이션을 사용하여 모든 파일 가져오기
                    dat_files = []
                    continuation_token = None
                    page_count = 0
                    
                    while True:
                        page_count += 1
                        if continuation_token:
                            response = self.s3_client.list_objects_v2(
                                Bucket=self.bucket_var.get(),
                                Prefix=prefix,
                                ContinuationToken=continuation_token
                            )
                        else:
                            response = self.s3_client.list_objects_v2(
                                Bucket=self.bucket_var.get(),
                                Prefix=prefix
                            )
                        
                        if 'Contents' in response:
                            files = response['Contents']
                            # .dat 파일만 필터링
                            page_dat_files = [f for f in files if f['Key'].endswith('.dat')]
                            dat_files.extend(page_dat_files)
                            self.log(f"  페이지 {page_count}: {len(files)}개 파일 중 {len(page_dat_files)}개 .dat 파일")
                        
                        # 다음 페이지가 있는지 확인
                        if response.get('IsTruncated', False):
                            continuation_token = response.get('NextContinuationToken')
                        else:
                            break
                    
                    self.log(f"  총 파일 수: {len(dat_files)}개")
                    
                    if dat_files:
                        # 날짜별 폴더 생성
                        date_folder = os.path.join(self.save_path_var.get(), 
                                                 machine_id, 
                                                 data_type.upper(),
                                                 current_date.strftime('%Y-%m-%d'))
                        os.makedirs(date_folder, exist_ok=True)
                        
                        # 병렬 다운로드 실행
                        self.log(f"  {len(dat_files)}개 파일 다운로드 시작 (스레드: {self.workers_var.get()}개)")
                        
                        downloaded_count, failed_count, elapsed_time, avg_speed = \
                            self.download_files_parallel(dat_files, date_folder, current_date)
                        
                        total_files += downloaded_count
                        
                        self.log(f"\n{current_date.strftime('%Y-%m-%d')} 완료:")
                        self.log(f"  - 성공: {downloaded_count}개")
                        if failed_count > 0:
                            self.log(f"  - 실패: {failed_count}개")
                        self.log(f"  - 소요 시간: {elapsed_time:.1f}초")
                        self.log(f"  - 평균 속도: {avg_speed:.1f} 파일/초")
                        self.log(f"  - 저장 위치: {date_folder}")
                    else:
                        self.log(f"{current_date.strftime('%Y-%m-%d')} - 데이터 없음")
                    
                except Exception as e:
                    self.log(f"오류 - {current_date.strftime('%Y-%m-%d')}: {str(e)}")
                
                # 진행률 업데이트
                processed_days += 1
                progress_value = (processed_days / total_days) * 100
                self.root.after(0, lambda v=progress_value: self.progress.__setitem__('value', v))
                
                current_date = current_date + timedelta(days=1)
            
            # 완료 메시지
            self.log(f"\n{'='*50}")
            self.log(f"다운로드 완료!")
            self.log(f"총 {total_files}개 파일 처리됨")
            self.log(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
            self.log(f"{'='*50}")
            
            self.root.after(0, lambda: self.status_label.config(text="다운로드 완료!"))
            self.root.after(0, lambda: messagebox.showinfo("완료", 
                f"모든 데이터 다운로드가 완료되었습니다.\n총 {total_files}개 파일 처리"))
            
        except Exception as e:
            error_msg = str(e)
            self.log(f"다운로드 중 오류 발생: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("오류", f"다운로드 중 오류 발생: {error_msg}"))
        finally:
            self.root.after(0, lambda: self.download_button.config(state='normal'))
    
    def format_file_size(self, size):
        """파일 크기를 읽기 쉬운 형식으로 변환"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def log(self, message):
        """로그 텍스트 영역에 메시지 추가"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = S3DataDownloader(root)
    root.mainloop()

if __name__ == "__main__":
    main()