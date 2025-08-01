#!/usr/bin/env python
# train_ocsvm_gui.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import json
import os
import threading
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
import optuna
from optuna import create_study
from tkcalendar import DateEntry
from tkcalendar import Calendar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# 라즈베리파이의 CustomRobustScaler 구현
class CustomRobustScaler:
    def __init__(self):
        self.params = {}
    
    def fit(self, X):
        self.params = {}
        for i in range(X.shape[1]):
            col = X[:, i]
            q1, median, q3 = np.percentile(col, [25, 50, 75])
            iqr = q3 - q1
            self.params[i] = {'median': median, 'iqr': iqr}
    
    def transform(self, X):
        X_scaled = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[1]):
            median = self.params[i]['median']
            iqr = self.params[i]['iqr']
            X_scaled[:, i] = (X[:, i] - median) / iqr
        return X_scaled
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def save(self, filepath):
        joblib.dump({"params": self.params}, filepath)
    
    def load(self, filepath):
        loaded = joblib.load(filepath)
        self.params = loaded["params"]
        
class OCSVMTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OCSVM 모델 학습 도구")
        self.root.geometry("1400x900")
        
        # 로그 텍스트 위젯을 먼저 초기화
        self.log_text = None
        
        # DB 연결
        self.conn = None
        
        # 설정값
        self.sensor_config = {
            "mic": {
                "sampling_rate": 8000,
                "window_sec": 5,
                "features": ["mav", "rms", "peak", "amp_iqr"],
                # Isolation Forest 파라미터
                "n_estimators_range": [100, 300],
                "contamination_range": [0.001, 0.05],
                "max_samples_range": ['auto', 0.5, 1.0],
                "max_features_range": [0.8, 1.0]
            },
            "acc": {
                "sampling_rate": 1666,
                "window_sec": 5,
                "features": ["x_peak", "x_crest_factor", "y_peak", "y_crest_factor", "z_peak", "z_crest_factor"],
                "n_estimators_range": [100, 300],
                "contamination_range": [0.001, 0.05],
                "max_samples_range": ['auto', 0.5, 1.0],
                "max_features_range": [0.8, 1.0]
            }
        }
        
        # 학습/테스트 기간 저장
        self.training_periods = []
        self.test_periods = []
        
        # GUI 생성
        self.create_widgets()
        
        # DB 연결 (GUI 생성 후)
        self.connect_db()
        
        # 날짜 범위 로드
        self.load_date_range()
    
    def log(self, message):
        """로그 메시지 출력"""
        if self.log_text:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.root.update()
        else:
            print(f"[LOG] {message}")  # 로그 위젯이 없을 때 콘솔 출력
        
        # 콘솔에도 항상 출력 (디버깅용)
        print(f"[{timestamp}] {message}")
    
    def connect_db(self):
        """DB 연결 - 트랜잭션 에러 방지"""
        try:
            self.conn = psycopg2.connect(
                host='localhost',
                port='5432',
                database='pdm_db',
                user='pdm_user',
                password='pdm_password'
            )
            # autocommit 설정으로 트랜잭션 에러 방지
            self.conn.autocommit = True
            self.log("✅ DB 연결 성공")
            
            # 테이블 및 데이터 확인
            cur = self.conn.cursor()
            
            # 테이블 확인
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('normal_acc_data', 'normal_mic_data')
            """)
            tables = [row[0] for row in cur.fetchall()]
            self.log(f"확인된 테이블: {tables}")
            
            # 각 테이블의 데이터 수 확인
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                self.log(f"{table}: {count:,}개 레코드")
            
            # 머신별 데이터 확인
            for table in tables:
                cur.execute(f"""
                    SELECT machine_id, COUNT(*) as cnt, 
                           MIN(time) as min_time, MAX(time) as max_time
                    FROM {table}
                    GROUP BY machine_id
                """)
                for row in cur.fetchall():
                    self.log(f"{table} - {row[0]}: {row[1]:,}개, {row[2]} ~ {row[3]}")
            
        except Exception as e:
            self.log(f"DB 연결 실패: {str(e)}")
            messagebox.showerror("DB 연결 실패", str(e))
    
    def create_widgets(self):
        # 메인 노트북
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 학습 탭
        train_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text="모델 학습")
        self.create_train_tab(train_tab)
        
        # 테스트 탭
        test_tab = ttk.Frame(notebook)
        notebook.add(test_tab, text="모델 테스트")
        self.create_test_tab(test_tab)
        
        # 로그 탭
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="로그")
        self.create_log_tab(log_tab)
    
    def create_train_tab(self, parent):
        # 설정 프레임
        config_frame = ttk.LabelFrame(parent, text="학습 설정", padding="10")
        config_frame.pack(fill='x', padx=5, pady=5)
        
        # 머신/센서 선택
        row = 0
        ttk.Label(config_frame, text="머신:").grid(row=row, column=0, sticky=tk.W, padx=5)
        self.machine_var = tk.StringVar(value="CURINGOVEN_M1")
        machine_combo = ttk.Combobox(config_frame, textvariable=self.machine_var, 
                                    values=["CURINGOVEN_M1", "HOTCHAMBER_M2"], width=20)
        machine_combo.grid(row=row, column=1, padx=5)
        
        ttk.Label(config_frame, text="센서:").grid(row=row, column=2, sticky=tk.W, padx=5)
        self.sensor_var = tk.StringVar(value="acc")
        sensor_combo = ttk.Combobox(config_frame, textvariable=self.sensor_var,
                                   values=["acc", "mic"], width=10)
        sensor_combo.grid(row=row, column=3, padx=5)
        
        # 최적화 설정
        row += 1
        ttk.Label(config_frame, text="Optuna Trials:").grid(row=row, column=0, sticky=tk.W, padx=5)
        self.trials_var = tk.IntVar(value=200)  # 더 많은 trials로 최적 gamma 탐색
        trials_spinbox = ttk.Spinbox(config_frame, from_=10, to=500, increment=10,
                                    textvariable=self.trials_var, width=10)
        trials_spinbox.grid(row=row, column=1, padx=5)
        
        # 성능 평가 스킵 옵션
        row += 1
        self.skip_eval_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="성능 평가 스킵 (빠른 학습)", 
                       variable=self.skip_eval_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        # 학습 기간 프레임
        period_frame = ttk.LabelFrame(parent, text="학습 기간 설정", padding="10")
        period_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 기간 입력
        input_frame = ttk.Frame(period_frame)
        input_frame.pack(fill='x')
        
        ttk.Label(input_frame, text="시작일:").pack(side='left', padx=5)
        self.train_start_date = DateEntry(input_frame, width=12, background='darkblue',
                                         foreground='white', borderwidth=2,
                                         date_pattern='yyyy-mm-dd')
        self.train_start_date.pack(side='left', padx=5)
        
        ttk.Label(input_frame, text="종료일:").pack(side='left', padx=5)
        self.train_end_date = DateEntry(input_frame, width=12, background='darkblue',
                                       foreground='white', borderwidth=2,
                                       date_pattern='yyyy-mm-dd')
        self.train_end_date.pack(side='left', padx=5)
        
        ttk.Button(input_frame, text="기간 추가", 
                  command=self.add_training_period).pack(side='left', padx=20)
        
        # 시간 필터 프레임 추가
        time_filter_frame = ttk.Frame(period_frame)
        time_filter_frame.pack(fill='x', pady=10)
        
        # 시간 필터 체크박스
        self.use_time_filter = tk.BooleanVar(value=False)
        self.time_filter_check = ttk.Checkbutton(time_filter_frame, text="시간 필터 사용", 
                                                 variable=self.use_time_filter,
                                                 command=self.toggle_time_filter)
        self.time_filter_check.pack(side='left', padx=5)
        
        # 시간 선택 위젯들
        self.time_widgets_frame = ttk.Frame(time_filter_frame)
        self.time_widgets_frame.pack(side='left', padx=20)
        
        ttk.Label(self.time_widgets_frame, text="시작 시간:").pack(side='left', padx=5)
        self.start_hour = ttk.Spinbox(self.time_widgets_frame, from_=0, to=23, width=3, format="%02.0f")
        self.start_hour.set("00")
        self.start_hour.pack(side='left')
        ttk.Label(self.time_widgets_frame, text=":").pack(side='left')
        self.start_minute = ttk.Spinbox(self.time_widgets_frame, from_=0, to=59, width=3, format="%02.0f")
        self.start_minute.set("00")
        self.start_minute.pack(side='left')
        
        ttk.Label(self.time_widgets_frame, text="  종료 시간:").pack(side='left', padx=(20,5))
        self.end_hour = ttk.Spinbox(self.time_widgets_frame, from_=0, to=23, width=3, format="%02.0f")
        self.end_hour.set("23")
        self.end_hour.pack(side='left')
        ttk.Label(self.time_widgets_frame, text=":").pack(side='left')
        self.end_minute = ttk.Spinbox(self.time_widgets_frame, from_=0, to=59, width=3, format="%02.0f")
        self.end_minute.set("59")
        self.end_minute.pack(side='left')
        
        # 기본적으로 시간 필터 비활성화
        self.toggle_time_filter()
        
        # 기간 목록
        list_frame = ttk.Frame(period_frame)
        list_frame.pack(fill='both', expand=True, pady=10)
        
        # 트리뷰
        columns = ('시작일', '종료일', '시간대', '일수')
        self.train_tree = ttk.Treeview(list_frame, columns=columns, height=8)
        self.train_tree.heading('#0', text='No')
        self.train_tree.heading('시작일', text='시작일')
        self.train_tree.heading('종료일', text='종료일')
        self.train_tree.heading('시간대', text='시간대')
        self.train_tree.heading('일수', text='일수')
        
        self.train_tree.column('#0', width=50)
        self.train_tree.column('시작일', width=150)
        self.train_tree.column('종료일', width=150)
        self.train_tree.column('시간대', width=100)
        self.train_tree.column('일수', width=80)
        
        self.train_tree.pack(side='left', fill='both', expand=True)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.train_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.train_tree.configure(yscrollcommand=scrollbar.set)
        
        # 버튼 프레임
        button_frame = ttk.Frame(period_frame)
        button_frame.pack(fill='x')
        
        ttk.Button(button_frame, text="선택 삭제", 
                  command=self.remove_training_period).pack(side='left', padx=5)
        ttk.Button(button_frame, text="모두 삭제", 
                  command=self.clear_training_periods).pack(side='left', padx=5)
        
        # 학습 버튼
        train_button_frame = ttk.Frame(parent)
        train_button_frame.pack(fill='x', padx=5, pady=10)
        
        self.train_button = ttk.Button(train_button_frame, text="모델 학습 시작", 
                                      command=self.start_training)
        self.train_button.pack(side='left', padx=5)
        
        self.progress_var = tk.StringVar(value="대기 중...")
        ttk.Label(train_button_frame, textvariable=self.progress_var).pack(side='left', padx=20)
    
    def create_test_tab(self, parent):
        # 모델 및 스케일러 선택 프레임
        model_frame = ttk.LabelFrame(parent, text="모델 및 스케일러 선택", padding="10")
        model_frame.pack(fill='x', padx=5, pady=5)
        
        # 센서 타입 선택
        ttk.Label(model_frame, text="센서 타입:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.test_sensor_var = tk.StringVar(value="acc")
        sensor_radio_frame = ttk.Frame(model_frame)
        sensor_radio_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=5)
        
        ttk.Radiobutton(sensor_radio_frame, text="가속도 센서", variable=self.test_sensor_var, 
                        value="acc", command=self.update_test_settings).pack(side='left', padx=5)
        ttk.Radiobutton(sensor_radio_frame, text="마이크 센서", variable=self.test_sensor_var, 
                        value="mic", command=self.update_test_settings).pack(side='left', padx=5)
        
        # 모델 파일 선택
        ttk.Label(model_frame, text="모델 파일:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.model_file_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_file_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(model_frame, text="찾아보기", 
                  command=self.browse_model_file).grid(row=1, column=2, padx=5)
        
        # 스케일러 파일 선택
        ttk.Label(model_frame, text="스케일러 파일:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.scaler_file_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.scaler_file_var, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(model_frame, text="찾아보기", 
                  command=self.browse_scaler_file).grid(row=2, column=2, padx=5)
        
        # 데이터베이스 설정 프레임
        db_frame = ttk.LabelFrame(parent, text="데이터베이스 설정", padding="10")
        db_frame.pack(fill='x', padx=5, pady=5)
        
        # 머신 선택
        ttk.Label(db_frame, text="머신:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.test_machine_var = tk.StringVar(value="CURINGOVEN_M1")
        ttk.Combobox(db_frame, textvariable=self.test_machine_var, 
                    values=["CURINGOVEN_M1", "HOTCHAMBER_M2"], 
                    width=20).grid(row=0, column=1, padx=5)
        
        # 샘플링 설정 표시
        ttk.Label(db_frame, text="샘플링 설정:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.sampling_info_var = tk.StringVar(value="가속도: 1666Hz, 5초 윈도우")
        ttk.Label(db_frame, textvariable=self.sampling_info_var).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # 특징 정보 표시
        ttk.Label(db_frame, text="추출 특징:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.features_info_var = tk.StringVar(value="x_peak, x_crest_factor, y_peak, y_crest_factor, z_peak, z_crest_factor")
        features_label = ttk.Label(db_frame, textvariable=self.features_info_var, wraplength=400)
        features_label.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # 테스트 기간 프레임
        test_period_frame = ttk.LabelFrame(parent, text="테스트 기간 설정", padding="10")
        test_period_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 기간 입력
        test_input_frame = ttk.Frame(test_period_frame)
        test_input_frame.pack(fill='x')
        
        ttk.Label(test_input_frame, text="시작일:").pack(side='left', padx=5)
        self.test_start_date = DateEntry(
            test_input_frame, 
            width=12, 
            background='darkblue',
            foreground='white', 
            borderwidth=2,
            date_pattern='yyyy-mm-dd',
            showweeknumbers=False,
            cursor="hand2"
        )
        self.test_start_date.pack(side='left', padx=5)
        
        ttk.Label(test_input_frame, text="종료일:").pack(side='left', padx=5)
        self.test_end_date = DateEntry(
            test_input_frame, 
            width=12, 
            background='darkblue',
            foreground='white', 
            borderwidth=2,
            date_pattern='yyyy-mm-dd',
            showweeknumbers=False,
            cursor="hand2"
        )
        self.test_end_date.pack(side='left', padx=5)
        
        ttk.Label(test_input_frame, text="(최대 5일)").pack(side='left', padx=10)
        
        # 테스트 버튼
        ttk.Button(parent, text="테스트 시작", 
                  command=self.start_testing).pack(pady=10)
        
        # 플롯 영역
        plot_frame = ttk.LabelFrame(parent, text="시계열 플롯", padding="10")
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # matplotlib Figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        test_button_frame = ttk.Frame(test_period_frame)
        test_button_frame.pack(fill='x')
        
        # 결과 표시
        result_frame = ttk.LabelFrame(parent, text="테스트 결과", padding="10")
        result_frame.pack(fill='x', padx=5, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=10)
        self.result_text.pack(fill='both', expand=True)
    
    def create_log_tab(self, parent):
        # 로그 텍스트 위젯 생성
        self.log_text = scrolledtext.ScrolledText(parent, height=30)
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def update_test_settings(self):
        """센서 타입에 따라 설정 정보 업데이트"""
        sensor = self.test_sensor_var.get()
        if sensor == "acc":
            self.sampling_info_var.set("가속도: 1666Hz, 5초 윈도우")
            self.features_info_var.set("x_peak, x_crest_factor, y_peak, y_crest_factor, z_peak, z_crest_factor")
        else:
            self.sampling_info_var.set("마이크: 8000Hz, 5초 윈도우")
            self.features_info_var.set("mav, rms, peak, amp_iqr")
    
    def browse_model_file(self):
        """모델 파일 찾아보기"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="모델 파일 선택",
            initialdir="./models",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.model_file_var.set(filename)
            
            # 모델 정보 파일도 같이 찾기
            info_path = filename.replace('_model.pkl', '_model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                    # 센서 타입 자동 설정
                    self.test_sensor_var.set(model_info.get('sensor', 'acc'))
                    self.update_test_settings()
    
    def browse_scaler_file(self):
        """스케일러 파일 찾아보기"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="스케일러 파일 선택",
            initialdir="./models",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.scaler_file_var.set(filename)
    
    def load_date_range(self):
        """DB에서 데이터 날짜 범위 확인"""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT MIN(DATE(time)), MAX(DATE(time))
                FROM normal_acc_data
            """)
            min_date, max_date = cur.fetchone()
            
            if min_date and max_date:
                self.train_start_date.set_date(min_date)
                self.train_end_date.set_date(max_date)
                self.test_start_date.set_date(max_date - timedelta(days=7))
                self.test_end_date.set_date(max_date)
                
                self.log(f"데이터 범위: {min_date} ~ {max_date}")
        except Exception as e:
            self.log(f"날짜 범위 로드 실패: {e}")
    
    def check_data_exists(self):
        """데이터 존재 여부 확인"""
        try:
            cur = self.conn.cursor()
            
            # 각 테이블의 데이터 수 확인
            for table in ['acc_data', 'mic_data']:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                self.log(f"{table}: {count}개 레코드")
                
            # 머신별 데이터 수 확인
            for table in ['acc_data', 'mic_data']:
                cur.execute(f"""
                    SELECT machine_id, COUNT(*), MIN(time), MAX(time)
                    FROM {table}
                    GROUP BY machine_id
                """)
                for row in cur.fetchall():
                    self.log(f"{table} - {row[0]}: {row[1]}개, {row[2]} ~ {row[3]}")
                    
        except Exception as e:
            self.log(f"데이터 확인 실패: {e}")
    
    def stratified_time_sampling(self, X_scaled, period_info, target_size):
        """시간대별 균등 샘플링"""
        sample_indices = []
        
        # 각 기간을 시간대별로 나누기
        for info in period_info:
            period_data_count = info['count']
            period_sample_size = int(target_size * (period_data_count / len(X_scaled)))
            
            if period_sample_size > 0:
                # 하루를 4개 시간대로 나누기 (6시간 단위)
                # 또는 피크/오프피크 시간대로 나누기
                period_start = info['start_idx']
                period_end = info['end_idx']
                
                # 균등 간격으로 샘플링
                indices = np.linspace(period_start, period_end-1, 
                                    period_sample_size, dtype=int)
                sample_indices.extend(indices)
        
        return np.array(sample_indices)
    
    def toggle_time_filter(self):
        """시간 필터 체크박스 상태에 따라 시간 선택 위젯 활성화/비활성화"""
        if self.use_time_filter.get():
            for widget in self.time_widgets_frame.winfo_children():
                widget.configure(state='normal')
        else:
            for widget in self.time_widgets_frame.winfo_children():
                if isinstance(widget, (ttk.Spinbox, ttk.Entry)):
                    widget.configure(state='disabled')
    
    def add_training_period(self):
        try:
            start_date = self.train_start_date.get_date()
            end_date = self.train_end_date.get_date()
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            
            if start_date > end_date:  # >= 에서 > 로 변경하여 같은 날 허용
                messagebox.showerror("오류", "시작일이 종료일보다 늦습니다.")
                return
            
            days = (end_date - start_date).days + 1
            
            # 시간 필터 정보 추가
            if self.use_time_filter.get():
                start_time = f"{self.start_hour.get()}:{self.start_minute.get()}"
                end_time = f"{self.end_hour.get()}:{self.end_minute.get()}"
                time_range = f"{start_time}~{end_time}"
            else:
                time_range = "전체"
                start_time = "00:00"
                end_time = "23:59"
            
            # 트리에 추가
            item_id = self.train_tree.insert('', 'end', 
                                           text=str(len(self.training_periods) + 1),
                                           values=(start, end, time_range, days))
            
            # 기간 정보에 시간 필터 포함
            period_info = {
                'start_date': start,
                'end_date': end,
                'start_time': start_time,
                'end_time': end_time,
                'use_time_filter': self.use_time_filter.get()
            }
            self.training_periods.append(period_info)
            self.log(f"학습 기간 추가: {start} ~ {end} ({time_range}) - {days}일")
            
        except ValueError:
            messagebox.showerror("오류", "날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)")
    
    def remove_training_period(self):
        selected = self.train_tree.selection()
        if selected:
            for item in selected:
                idx = self.train_tree.index(item)
                del self.training_periods[idx]
                self.train_tree.delete(item)
            
            # 번호 재정렬
            for i, item in enumerate(self.train_tree.get_children()):
                self.train_tree.item(item, text=str(i + 1))
    
    def clear_training_periods(self):
        self.training_periods.clear()
        for item in self.train_tree.get_children():
            self.train_tree.delete(item)

    
    def extract_features_acc(self, x_data, y_data, z_data):
        """ACC 특징 추출 - 다운샘플링된 데이터용"""
        try:
            # 데이터가 너무 적으면 스킵
            if len(x_data) < 10:
                raise ValueError(f"데이터가 너무 적음: {len(x_data)}개")
            
            features = []
            
            # X축
            x_rms = np.sqrt(np.mean(x_data**2))
            x_peak = np.max(np.abs(x_data))
            x_crest = x_peak / x_rms if x_rms > 1e-10 else 0
            
            # Y축
            y_rms = np.sqrt(np.mean(y_data**2))
            y_peak = np.max(np.abs(y_data))
            y_crest = y_peak / y_rms if y_rms > 1e-10 else 0
            
            # Z축
            z_rms = np.sqrt(np.mean(z_data**2))
            z_peak = np.max(np.abs(z_data))
            z_crest = z_peak / z_rms if z_rms > 1e-10 else 0
            
            return np.array([x_peak, x_crest, y_peak, y_crest, z_peak, z_crest])
            
        except Exception as e:
            self.log(f"ACC 특징 추출 오류: {e}")
            raise
    
    def extract_features_mic(self, mic_data):
        """MIC 특징 추출 - 다운샘플링된 데이터용"""
        try:
            # 데이터가 너무 적으면 스킵
            if len(mic_data) < 10:
                raise ValueError(f"데이터가 너무 적음: {len(mic_data)}개")
            
            mav = np.mean(np.abs(mic_data))
            rms = np.sqrt(np.mean(mic_data**2))
            peak = np.max(np.abs(mic_data))
            
            # IQR 계산 시 데이터가 적으므로 조심
            q1, q3 = np.percentile(np.abs(mic_data), [25, 75])
            amp_iqr = q3 - q1
            
            return np.array([mav, rms, peak, amp_iqr])
            
        except Exception as e:
            self.log(f"MIC 특징 추출 오류: {e}")
            raise
    
    def get_training_data(self, machine_id, sensor, start_date, end_date, start_time="00:00", end_time="23:59", return_timestamps=False):
        """DB에서 학습 데이터 추출 - 다운샘플링된 데이터 처리"""
        window_sec = self.sensor_config[sensor]['window_sec']
        
        # DB에는 1초에 10개씩만 저장되어 있음
        db_sampling_rate = 10  # 1초에 10개
        window_samples = window_sec * db_sampling_rate  # 5초 * 10 = 50개
        
        # 전체 데이터를 가져옴
        if sensor == 'acc':
            query = """
            SELECT time, x, y, z
            FROM normal_acc_data
            WHERE machine_id = %s
            AND time >= %s::date + %s::time 
            AND time <= %s::date + %s::time
            ORDER BY time
            """
        else:  # mic
            query = """
            SELECT time, mic_value
            FROM normal_mic_data
            WHERE machine_id = %s
            AND time >= %s::date + %s::time 
            AND time <= %s::date + %s::time
            ORDER BY time
            """
        
        self.log(f"데이터 추출 중: {machine_id}, {sensor}, {start_date} {start_time} ~ {end_date} {end_time}")
        
        try:
            start_time_obj = datetime.now()
            df = pd.read_sql(query, self.conn, 
                            params=(machine_id, start_date, start_time, end_date, end_time))
            
            if df.empty:
                self.log(f"데이터가 없습니다!")
                return None
            
            self.log(f"전체 데이터: {len(df)}개 샘플")
            self.log(f"데이터 로드 시간: {(datetime.now() - start_time_obj).total_seconds():.1f}초")
            
            # Python에서 5초 윈도우로 분할
            features_list = []
            timestamps_list = []  # 윈도우 타임스탬프 저장
            
            # 시간을 datetime으로 변환
            df['time'] = pd.to_datetime(df['time'])
            
            # 5초 단위로 그룹화
            df['window'] = df['time'].dt.floor(f'{window_sec}s')  # 'S' -> 's'로 변경
            
            # 윈도우별로 처리
            window_count = 0
            for window, group in df.groupby('window'):
                # 최소 40개 이상 (80%) 데이터가 있는 윈도우만 사용
                if len(group) >= window_samples * 0.8:  
                    try:
                        if sensor == 'acc':
                            features = self.extract_features_acc(
                                group['x'].values,
                                group['y'].values,
                                group['z'].values
                            )
                        else:
                            features = self.extract_features_mic(group['mic_value'].values)
                        
                        features_list.append(features)
                        window_count += 1
                        
                        # 타임스탬프 저장 (윈도우의 시작 시간)
                        if return_timestamps:
                            timestamps_list.append(window)
                        
                        # 진행 상황 로그 (1000개마다)
                        if window_count % 1000 == 0:
                            elapsed = (datetime.now() - start_time_obj).total_seconds()
                            rate = window_count / elapsed if elapsed > 0 else 0
                            self.log(f"  처리중: {window_count:,}개 윈도우 ({rate:.0f} windows/sec)")
                        
                        # GUI 업데이트 (100개마다)
                        if window_count % 100 == 0:
                            self.progress_var.set(f"데이터 추출 중: {start_date} ~ {end_date} ({window_count}개)")
                            self.root.update_idletasks()
                            
                    except Exception as e:
                        if window_count % 1000 == 0:
                            self.log(f"  윈도우 처리 오류 발생: {e}")
                        continue
            
            total_time = (datetime.now() - start_time_obj).total_seconds()
            self.log(f"추출 완료: {len(features_list)}개 윈도우 (총 {total_time:.1f}초)")
            
            if return_timestamps:
                return np.array(features_list) if features_list else None, timestamps_list
            else:
                return np.array(features_list) if features_list else None
            
        except Exception as e:
            self.log(f"데이터 추출 오류: {e}")
            return None
    
    def train_model(self):
        """모델 학습 (별도 스레드에서 실행)"""
        try:
            total_start_time = datetime.now()
            machine_id = self.machine_var.get()
            sensor = self.sensor_var.get()
            n_trials = self.trials_var.get()
            
            self.log(f"\n{'='*60}")
            self.log(f"{machine_id} / {sensor} 모델 학습 시작")
            self.log(f"{'='*60}")
            
            # 학습 데이터 수집
            all_features = []
            period_info = []  # 각 기간별 정보 저장
            
            self.log(f"학습 기간: {len(self.training_periods)}개")
            
            for idx, period in enumerate(self.training_periods):
                # 기간 정보 추출 (이전 버전 호환성 유지)
                if isinstance(period, dict):
                    start = period['start_date']
                    end = period['end_date']
                    start_time = period.get('start_time', '00:00')
                    end_time = period.get('end_time', '23:59')
                    time_str = f" ({start_time}~{end_time})" if period.get('use_time_filter', False) else ""
                else:
                    # 이전 버전 호환 (튜플 형식)
                    start, end = period
                    start_time, end_time = '00:00', '23:59'
                    time_str = ""
                
                self.log(f"\n[{idx+1}/{len(self.training_periods)}] 기간: {start} ~ {end}{time_str}")
                self.progress_var.set(f"데이터 추출 중: {start} ~ {end}{time_str}")
                features = self.get_training_data(machine_id, sensor, start, end, start_time, end_time)
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    period_info.append({
                        'period': f"{start} ~ {end}{time_str}",
                        'start_idx': len(np.vstack(all_features[:-1])) if len(all_features) > 1 else 0,
                        'end_idx': len(np.vstack(all_features)),
                        'count': len(features)
                    })
                    self.log(f"✅ 추출 성공: {len(features)}개 윈도우")
            
            if not all_features:
                self.log("❌ 학습 데이터가 없습니다.")
                self.progress_var.set("학습 데이터 없음")
                return
            
            X_train = np.vstack(all_features)
            self.log(f"\n전체 학습 데이터: {X_train.shape}")
            
            # 스케일러 학습 (fit만 수행)
            self.log("\n스케일러 학습 시작...")
            self.progress_var.set("스케일러 학습 중...")
            scaler_start = datetime.now()
            scaler = CustomRobustScaler()
            scaler.fit(X_train)  # 전체 데이터로 범위만 학습
            scaler_time = (datetime.now() - scaler_start).total_seconds()
            self.log(f"✅ 스케일러 학습 완료 ({scaler_time:.1f}초)")
            
            # 전체 데이터 스케일링 (한 번에)
            self.log("\n전체 데이터 스케일링 시작...")
            self.progress_var.set("데이터 스케일링 중...")
            scaling_start = datetime.now()
            X_scaled = scaler.transform(X_train)  # 전체를 한 번에 변환
            scaling_time = (datetime.now() - scaling_start).total_seconds()
            self.log(f"✅ 데이터 스케일링 완료 ({scaling_time:.1f}초)")
            
            # 🔍 디버깅: 원본 데이터와 스케일된 데이터 비교
            self.log("\n🔍 [디버깅] 데이터 스케일링 검증:")
            for i, feature_name in enumerate(self.sensor_config[sensor]['features']):
                self.log(f"\n  [{feature_name}]")
                self.log(f"    원본 - min: {X_train[:, i].min():.2f}, max: {X_train[:, i].max():.2f}, "
                        f"mean: {X_train[:, i].mean():.2f}, std: {X_train[:, i].std():.2f}")
                self.log(f"    스케일 - min: {X_scaled[:, i].min():.2f}, max: {X_scaled[:, i].max():.2f}, "
                        f"mean: {X_scaled[:, i].mean():.2f}, std: {X_scaled[:, i].std():.2f}")
                
                # 스케일러 파라미터 확인
                self.log(f"    스케일러 - median: {scaler.params[i]['median']:.2f}, "
                        f"IQR: {scaler.params[i]['iqr']:.2f}")
            
            # 🔍 전체 스케일된 데이터 통계
            self.log(f"\n🔍 [디버깅] 전체 스케일된 데이터:")
            self.log(f"  - 범위: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
            self.log(f"  - 평균: {X_scaled.mean():.4f}")
            self.log(f"  - 표준편차: {X_scaled.std():.4f}")
            self.log(f"  - 중앙값: {np.median(X_scaled):.4f}")
            self.log(f"  - 95% 범위: [{np.percentile(X_scaled, 2.5):.4f}, "
                    f"{np.percentile(X_scaled, 97.5):.4f}]")
            

            # 역변환 함수 (테스트용)
            def inverse_transform_scores(transformed_scores, transform_info):
                """변환된 점수를 원본으로 역변환 (exponential transform의 역변환은 복잡하므로 근사치 사용)"""
                # 새로운 exponential transform의 정확한 역변환은 복잡하므로
                # 근사적인 역변환을 제공 (주로 디버깅/테스트 목적)
                median_score = transform_info.get('median_score', -0.5)
                return transformed_scores * 0.1 + median_score  # 간단한 근사
            
            # Isolation Forest 최적화
            self.log(f"\n하이퍼파라미터 최적화 시작 (Optuna {n_trials} trials)")
            optuna_start = datetime.now()
            self.progress_var.set(f"하이퍼파라미터 최적화 중... (0/{n_trials})")
            
            opt_config = self.sensor_config[sensor]
            n_estimators_range = opt_config['n_estimators_range']
            contamination_range = opt_config['contamination_range']
            max_samples_range = opt_config['max_samples_range']
            max_features_range = opt_config['max_features_range']
            
            # 시간적 분포를 고려한 계층적 샘플링
            target_sample_size = min(20000, int(len(X_scaled) * 0.1))  # 최대 20,000개
            
            if target_sample_size < len(X_scaled):
                self.log(f"\n📊 계층적 샘플링 시작: {len(X_scaled)}개 → {target_sample_size}개")
                
                sample_indices = []
                
                # 각 기간별로 비례적으로 샘플링
                for info in period_info:
                    period_ratio = info['count'] / len(X_scaled)
                    period_sample_size = int(target_sample_size * period_ratio)
                    
                    if period_sample_size > 0:
                        # 해당 기간 내에서 균등하게 샘플링
                        period_indices = np.arange(info['start_idx'], info['end_idx'])
                        
                        if period_sample_size < len(period_indices):
                            # 시간 간격을 두고 균등하게 선택
                            step = len(period_indices) // period_sample_size
                            selected = period_indices[::step][:period_sample_size]
                        else:
                            selected = period_indices
                        
                        sample_indices.extend(selected)
                        self.log(f"  - {info['period']}: {len(selected)}개 샘플")
                
                sample_indices = np.array(sample_indices)
                
                # 샘플링된 데이터 스케일링
                X_sample = scaler.transform(X_train[sample_indices])
                self.log(f"✅ 총 {len(X_sample)}개 샘플 추출 완료")
            else:
                X_sample = X_scaled
                self.log(f"📊 데이터가 충분히 작아 전체 사용: {len(X_scaled)}개")
            
            trial_count = 0
            # study 변수를 클래스 변수로 만들어 objective 함수에서 접근 가능하게
            self.study = create_study(direction='maximize')
            
            # K-fold 설정 (라즈베리파이는 3을 사용)
            n_splits = 3
            
            def objective(trial):
                # 진행률 업데이트
                current_trial = len(self.study.trials)
                self.progress_var.set(f"하이퍼파라미터 최적화 중... ({current_trial}/{n_trials})")
                self.root.update_idletasks()
                
                # Isolation Forest 하이퍼파라미터
                n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1])
                contamination = trial.suggest_float('contamination', contamination_range[0], contamination_range[1], log=True)
                
                # max_samples 처리
                if max_samples_range[0] == 'auto':
                    max_samples = trial.suggest_categorical('max_samples', ['auto'] + list(max_samples_range[1:]))
                else:
                    max_samples = trial.suggest_float('max_samples', max_samples_range[0], max_samples_range[-1])
                
                max_features = trial.suggest_float('max_features', max_features_range[0], max_features_range[1])
                
                # K-Fold 사용 (라즈베리파이와 동일)
                if n_splits <= 1:
                    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                          max_samples=max_samples, max_features=max_features,
                                          random_state=42, n_jobs=-1)
                    model.fit(X_sample)
                    preds = model.predict(X_sample)
                    return np.mean(preds == -1)  # 이상치 비율
                
                # K-Fold가 있는 경우
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = []
                for train_idx, test_idx in kf.split(X_sample):
                    X_train, X_test = X_sample[train_idx], X_sample[test_idx]
                    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                          max_samples=max_samples, max_features=max_features,
                                          random_state=42, n_jobs=-1)
                    model.fit(X_train)
                    preds = model.predict(X_test)
                    scores.append(np.mean(preds == -1))
                
                return np.mean(scores)
            
            # Optuna 콜백 함수
            def optuna_callback(study, trial):
                nonlocal trial_count
                trial_count += 1
                if trial_count % 10 == 0 or trial_count <= 5:
                    max_samples_str = trial.params.get('max_samples', 'N/A')
                    if isinstance(max_samples_str, float):
                        max_samples_str = f"{max_samples_str:.2f}"
                    
                    self.log(f"  Trial {trial_count}: n_estimators={trial.params['n_estimators']}, "
                            f"contamination={trial.params['contamination']:.4f}, "
                            f"max_samples={max_samples_str}, score={trial.value:.4f}")
            
            # 최적화 실행 (direction='minimize'로 변경)
            self.study = create_study(direction='minimize')  # 이상치 비율 최소화
            self.study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
            
            optuna_time = (datetime.now() - optuna_start).total_seconds()
            self.log(f"\n✅ 최적화 완료 ({optuna_time:.1f}초)")
            
            best_params_str = f"n_estimators={self.study.best_params['n_estimators']}, "
            best_params_str += f"contamination={self.study.best_params['contamination']:.4f}, "
            best_params_str += f"max_samples={self.study.best_params.get('max_samples', 'auto')}, "
            best_params_str += f"max_features={self.study.best_params['max_features']:.2f}"
            
            self.log(f"최적 파라미터: {best_params_str}")
            self.log(f"최적 점수: {self.study.best_value:.4f}")
            
            # 최적 모델로 전체 데이터 학습
            self.progress_var.set("최종 모델 학습 중...")
            self.log("\n🔍 최종 모델 학습 데이터 확인...")
            best_n_estimators = self.study.best_params['n_estimators']
            best_contamination = self.study.best_params['contamination']
            best_max_samples = self.study.best_params.get('max_samples', 'auto')
            best_max_features = self.study.best_params['max_features']
            
            # 학습 직전 데이터 확인
            self.log(f"\n🔍 [중요] 최종 학습 데이터 검증:")
            self.log(f"  - X_scaled shape: {X_scaled.shape}")
            self.log(f"  - X_scaled dtype: {X_scaled.dtype}")
            self.log(f"  - X_scaled 전체 범위: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
            self.log(f"  - X_scaled 전체 평균: {X_scaled.mean():.4f}")
            self.log(f"  - X_scaled 전체 표준편차: {X_scaled.std():.4f}")
            
            # 첫 5개 샘플 상세 확인
            self.log(f"\n🔍 첫 5개 샘플 상세 확인:")
            for i in range(min(5, len(X_scaled))):
                self.log(f"  샘플 {i}: {X_scaled[i]}")
            
            # 각 특징별 범위 확인
            self.log(f"\n🔍 각 특징별 스케일된 범위:")
            for i, feature_name in enumerate(self.sensor_config[sensor]['features']):
                self.log(f"  [{feature_name}] 범위: [{X_scaled[:, i].min():.4f}, {X_scaled[:, i].max():.4f}], "
                        f"평균: {X_scaled[:, i].mean():.4f}, 표준편차: {X_scaled[:, i].std():.4f}")
            
            model = IsolationForest(n_estimators=best_n_estimators, contamination=best_contamination,
                                  max_samples=best_max_samples, max_features=best_max_features,
                                  random_state=42, n_jobs=-1)
            
            self.log(f"\n최종 모델 학습 시작 ({best_params_str})...")
            
            # 대규모 데이터 처리: 기간별 비례 샘플링
            max_train_samples = 10000
            if len(X_scaled) > max_train_samples:
                self.log(f"\n📊 기간별 비례 샘플링: {len(X_scaled):,}개 → {max_train_samples:,}개")
                
                sampled_indices = []
                for info in period_info:
                    period_start = info['start_idx']
                    period_end = info['end_idx']
                    period_count = period_end - period_start
                    
                    # 각 기간에서 비례적으로 샘플
                    period_sample_size = int(max_train_samples * (period_count / len(X_scaled)))
                    if period_sample_size > 0:
                        # 균등 간격으로 샘플링
                        indices = np.linspace(period_start, period_end-1, period_sample_size, dtype=int)
                        sampled_indices.extend(indices)
                        self.log(f"  - {info['period']}: {period_sample_size}개 샘플")
                
                sampled_indices = np.array(sampled_indices)
                X_train_final = X_scaled[sampled_indices]
            else:
                X_train_final = X_scaled
                
            model.fit(X_train_final)
            
            self.log("✅ 최종 모델 학습 완료")
            
            # 🔍 디버깅: Isolation Forest 모델 정보
            self.log(f"\n🔍 [디버깅] Isolation Forest 모델 정보:")
            self.log(f"  - 트리 개수: {len(model.estimators_)}개")
            self.log(f"  - 샘플 크기: {model.max_samples_}")
            if hasattr(model, 'offset_'):
                self.log(f"  - Offset: {model.offset_}")
            
            # 모델 성능 평가 (선택적)
            skip_evaluation = self.skip_eval_var.get()  # GUI 체크박스 값 사용
            
            # 🔍 항상 작은 샘플로 score 분포 확인
            sample_size = min(1000, len(X_scaled))
            sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            debug_scores = model.score_samples(X_scaled[sample_indices])
            
            # 점수 변환
            debug_transformed, transform_info = self.transform_scores(debug_scores)
            
            self.log(f"\n🔍 [디버깅] 샘플 {sample_size}개의 score 분포 (원본):")
            self.log(f"  - 범위: [{debug_scores.min():.2f}, {debug_scores.max():.2f}]")
            self.log(f"  - 변환 후: [{debug_transformed.min():.2f}, {debug_transformed.max():.2f}]")
            
            if skip_evaluation:
                self.log("\n⚡ 성능 평가 단계 스킵 (빠른 학습 모드)")
                self.log(f"전체 데이터({len(X_scaled):,}개)로 score 계산 중...")
                
                # 간단한 샘플링으로 대략적인 성능만 확인
                sample_size = min(10000, len(X_scaled))
                sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
                sample_scores = model.score_samples(X_scaled[sample_indices])
                
                # 점수 변환
                sample_scores_transformed, transform_info = self.transform_scores(sample_scores)
                
                # 결정 경계 설정: 정상 데이터의 하위 5%를 경계로
                decision_boundary = np.percentile(sample_scores_transformed, 5)
                
                # 🔍 디버깅: boundary 계산 과정
                self.log(f"\n🔍 [디버깅] Decision Boundary 계산:")
                self.log(f"  - 원본 Score 분포: [{sample_scores.min():.4f}, {sample_scores.max():.4f}]")
                self.log(f"  - 변환 Score 분포: [{sample_scores_transformed.min():.2f}, {sample_scores_transformed.max():.2f}]")
                self.log(f"  - 결정 경계: {decision_boundary:.3f}")
                
                # 정상 데이터 분포 확인
                normal_scores = sample_scores_transformed[sample_scores_transformed > decision_boundary]
                self.log(f"\n📊 정상 데이터 분포:")
                self.log(f"  - 범위: [{normal_scores.min():.3f}, {normal_scores.max():.3f}]")
                self.log(f"  - 평균: {normal_scores.mean():.3f}")
                self.log(f"  - 중앙값: {np.median(normal_scores):.3f}")
                
                model_info = {
                    'machine_id': machine_id,
                    'sensor': sensor,
                    'model_type': 'IsolationForest',
                    'train_samples': len(X_train),
                    'training_periods': self.training_periods,
                    'features': self.sensor_config[sensor]['features'],
                    'best_params': self.study.best_params,
                    'decision_boundary': float(decision_boundary),
                    'boundary_method': 'percentile_5',
                    'score_transform': transform_info,
                    'boundary_stats': {
                        'percentile_5': float(decision_boundary),
                        'method': 'percentile_based'
                    },
                    'evaluation_skipped': True,
                    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                # 전체 성능 평가 (배치 처리)
                self.log("\n모델 성능 평가 중...")
                eval_start = datetime.now()
                
                batch_size = 10000
                predictions = []
                scores = []
                
                self.log(f"전체 {len(X_scaled):,}개 데이터에 대해 예측 수행 (배치 크기: {batch_size:,})")
                
                for i in range(0, len(X_scaled), batch_size):
                    batch_end = min(i + batch_size, len(X_scaled))
                    batch = X_scaled[i:batch_end]
                    
                    # 배치 예측
                    batch_predictions = model.predict(batch)
                    batch_scores = model.score_samples(batch)
                    
                    predictions.extend(batch_predictions)
                    scores.extend(batch_scores)
                    
                    # 진행 상황 로그 (10개 배치마다)
                    if (i // batch_size + 1) % 10 == 0 or batch_end == len(X_scaled):
                        progress = batch_end / len(X_scaled) * 100
                        elapsed = (datetime.now() - eval_start).total_seconds()
                        rate = batch_end / elapsed if elapsed > 0 else 0
                        eta = (len(X_scaled) - batch_end) / rate if rate > 0 else 0
                        
                        self.log(f"  예측 진행: {batch_end:,}/{len(X_scaled):,} ({progress:.1f}%) "
                                f"- {rate:.0f} samples/sec, ETA: {eta:.0f}초")
                        self.progress_var.set(f"성능 평가 중... {progress:.1f}%")
                        self.root.update_idletasks()
                
                # numpy 배열로 변환
                predictions = np.array(predictions)
                scores = np.array(scores)
                
                eval_time = (datetime.now() - eval_start).total_seconds()
                self.log(f"✅ 예측 완료 ({eval_time:.1f}초)")
                
                # 점수 변환
                scores_transformed, transform_info = self.transform_scores(scores)
                
                # 결정 경계 계산 (5% 백분위수 방식)
                self.log("\n결정 경계 계산 중...")
                
                # 결정 경계 설정: 정상 데이터의 하위 5%를 경계로
                decision_boundary = np.percentile(scores_transformed, 5)
                
                # 정상 데이터 분포 확인
                normal_scores_transformed = scores_transformed[scores_transformed > decision_boundary]
                self.log(f"\n📊 정상 데이터 분포:")
                self.log(f"  - 범위: [{normal_scores_transformed.min():.3f}, {normal_scores_transformed.max():.3f}]")
                self.log(f"  - 평균: {normal_scores_transformed.mean():.3f}")
                self.log(f"  - 중앙값: {np.median(normal_scores_transformed):.3f}")
                self.log(f"  - 결정 경계: {decision_boundary:.3f}")
                
                # 🔍 디버깅: 전체 평가에서도 경계값 확인
                self.log(f"\n🔍 [디버깅] 전체 평가 Decision Boundary:")
                self.log(f"  - 전체 Score 분포 (원본):")
                self.log(f"    • 범위: [{np.min(scores):.2f}, {np.max(scores):.2f}]")
                self.log(f"    • 평균: {np.mean(scores):.2f}")
                self.log(f"    • 표준편차: {np.std(scores):.2f}")
                self.log(f"  - 전체 Score 분포 (변환):")
                self.log(f"    • 범위: [{np.min(scores_transformed):.2f}, {np.max(scores_transformed):.2f}]")
                self.log(f"    • 평균: {np.mean(scores_transformed):.2f}")
                self.log(f"    • 표준편차: {np.std(scores_transformed):.2f}")
                self.log(f"  - Percentiles:")
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                    self.log(f"    • P{p}: {np.percentile(scores_transformed, p):.2f}")
                
                
                # 이상치 비율 계산
                predictions = (scores_transformed > decision_boundary).astype(int) * 2 - 1
                anomaly_ratio = np.sum(predictions == -1) / len(predictions) * 100
                
                self.log(f"✅ 학습 완료!")
                self.log(f"  - 이상치 비율: {anomaly_ratio:.2f}%")
                self.log(f"  - 결정 경계: {decision_boundary:.4f}")
                self.log(f"  - 변환 점수 범위: [{np.min(scores_transformed):.4f}, {np.max(scores_transformed):.4f}]")
                self.log(f"  - 변환 점수 평균±표준편차: {np.mean(scores_transformed):.4f} ± {np.std(scores_transformed):.4f}")
                
                # 기간별 성능 분석
                self.log("\n📊 기간별 성능:")
                for info in period_info:
                    start_idx = info['start_idx']
                    end_idx = info['end_idx']
                    period_scores_transformed = scores_transformed[start_idx:end_idx]
                    period_predictions = predictions[start_idx:end_idx]
                    period_anomaly_ratio = np.sum(period_predictions == -1) / len(period_predictions) * 100
                    
                    self.log(f"  - {info['period']}: 이상 {period_anomaly_ratio:.1f}%, "
                            f"점수 {np.mean(period_scores_transformed):.2f}±{np.std(period_scores_transformed):.2f}")
                
                # 2차 로직을 위한 상세 통계 분석
                self.log("\n📊 2차 로직 경계값 설정을 위한 분석:")
                
                # 정상/이상 데이터 분리
                normal_scores_transformed = scores_transformed[predictions == 1]
                anomaly_scores_transformed = scores_transformed[predictions == -1]
                
                self.log(f"  정상 데이터 점수 분포:")
                self.log(f"    - 개수: {len(normal_scores_transformed):,}개 ({len(normal_scores_transformed)/len(scores_transformed)*100:.1f}%)")
                self.log(f"    - 평균±표준편차: {np.mean(normal_scores_transformed):.2f} ± {np.std(normal_scores_transformed):.2f}")
                self.log(f"    - 최소/최대: {np.min(normal_scores_transformed):.2f} / {np.max(normal_scores_transformed):.2f}")
                
                self.log(f"  이상 데이터 점수 분포:")
                self.log(f"    - 개수: {len(anomaly_scores_transformed):,}개 ({len(anomaly_scores_transformed)/len(scores_transformed)*100:.1f}%)")
                self.log(f"    - 평균±표준편차: {np.mean(anomaly_scores_transformed):.2f} ± {np.std(anomaly_scores_transformed):.2f}")
                self.log(f"    - 최소/최대: {np.min(anomaly_scores_transformed):.2f} / {np.max(anomaly_scores_transformed):.2f}")
                
                # 퍼센타일 기반 경계값 후보
                percentiles = [0.1, 0.5, 1, 2, 3, 5, 10, 15, 20]
                percentile_values = {}
                
                self.log("\n  전체 점수 퍼센타일:")
                for p in percentiles:
                    val = np.percentile(scores_transformed, p)
                    percentile_values[f"p{p}"] = float(val)
                    self.log(f"    - {p:5.1f}%: {val:8.2f}")
                
                # 점수 구간별 분포
                self.log("\n  점수 구간별 분포:")
                score_ranges = [
                    (-np.inf, -17, "극심한 이상"),
                    (-17, -15, "심각한 이상"),
                    (-15, -10, "중간 이상"),
                    (-10, -5, "경미한 이상"),
                    (-5, -2, "의심 구간"),
                    (-2, 0, "경계 구간"),
                    (0, 1, "정상 범위"),
                    (1, np.inf, "매우 정상")
                ]
                
                score_distribution = {}
                self.log("    [전체 데이터]")
                for min_score, max_score, label in score_ranges:
                    count = np.sum((scores_transformed >= min_score) & (scores_transformed < max_score))
                    ratio = count / len(scores_transformed) * 100
                    self.log(f"    - {label:12s} [{min_score:6.0f} ~ {max_score:6.0f}]: "
                            f"{count:6,}개 ({ratio:5.1f}%)")
                
                # 정상 데이터의 점수 구간별 분포
                self.log("\n    [정상으로 분류된 데이터]")
                normal_distribution = {}
                for min_score, max_score, label in score_ranges:
                    count = np.sum((normal_scores_transformed >= min_score) & (normal_scores_transformed < max_score))
                    ratio = count / len(normal_scores_transformed) * 100 if len(normal_scores_transformed) > 0 else 0
                    normal_distribution[label] = {
                        'count': int(count),
                        'ratio': float(ratio),
                        'range': [float(min_score) if min_score != -np.inf else None,
                                 float(max_score) if max_score != np.inf else None]
                    }
                    if count > 0:
                        self.log(f"    - {label:12s} [{min_score:6.0f} ~ {max_score:6.0f}]: "
                                f"{count:6,}개 ({ratio:5.1f}%)")
                
                # 이상 데이터의 점수 구간별 분포
                self.log("\n    [이상으로 분류된 데이터]")
                anomaly_distribution = {}
                for min_score, max_score, label in score_ranges:
                    count = np.sum((anomaly_scores_transformed >= min_score) & (anomaly_scores_transformed < max_score))
                    ratio = count / len(anomaly_scores_transformed) * 100 if len(anomaly_scores_transformed) > 0 else 0
                    anomaly_distribution[label] = {
                        'count': int(count),
                        'ratio': float(ratio),
                        'range': [float(min_score) if min_score != -np.inf else None,
                                 float(max_score) if max_score != np.inf else None]
                    }
                    if count > 0:
                        self.log(f"    - {label:12s} [{min_score:6.0f} ~ {max_score:6.0f}]: "
                                f"{count:6,}개 ({ratio:5.1f}%)")
                
                # 전체 통합 분포
                for min_score, max_score, label in score_ranges:
                    total_count = np.sum((scores_transformed >= min_score) & (scores_transformed < max_score))
                    normal_count = np.sum((normal_scores_transformed >= min_score) & (normal_scores_transformed < max_score))
                    anomaly_count = np.sum((anomaly_scores_transformed >= min_score) & (anomaly_scores_transformed < max_score))
                    
                    score_distribution[label] = {
                        'total': {
                            'count': int(total_count),
                            'ratio': float(total_count / len(scores_transformed) * 100)
                        },
                        'normal': {
                            'count': int(normal_count),
                            'ratio': float(normal_count / len(normal_scores_transformed) * 100) if len(normal_scores_transformed) > 0 else 0,
                            'of_total': float(normal_count / total_count * 100) if total_count > 0 else 0
                        },
                        'anomaly': {
                            'count': int(anomaly_count),
                            'ratio': float(anomaly_count / len(anomaly_scores_transformed) * 100) if len(anomaly_scores_transformed) > 0 else 0,
                            'of_total': float(anomaly_count / total_count * 100) if total_count > 0 else 0
                        },
                        'range': [float(min_score) if min_score != -np.inf else None,
                                 float(max_score) if max_score != np.inf else None]
                    }
                
                # 교차 분석
                self.log("\n  📊 정상/이상 교차 분석:")
                
                # 정상으로 분류되었지만 점수가 낮은 데이터
                normal_but_low_score = np.sum(normal_scores_transformed < -2)
                if normal_but_low_score > 0:
                    self.log(f"    - 정상 분류지만 점수 < -2: {normal_but_low_score:,}개 "
                            f"({normal_but_low_score/len(normal_scores_transformed)*100:.1f}%)")
                    
                    # 상세 분포
                    for threshold in [-5, -10, -15]:
                        count = np.sum(normal_scores_transformed < threshold)
                        if count > 0:
                            self.log(f"      • 점수 < {threshold}: {count:,}개 "
                                    f"({count/len(normal_scores_transformed)*100:.2f}%)")
                
                # 이상으로 분류되었지만 점수가 높은 데이터
                if len(anomaly_scores_transformed) > 0:
                    anomaly_but_high_score = np.sum(anomaly_scores_transformed > 0)
                    if anomaly_but_high_score > 0:
                        self.log(f"    - 이상 분류지만 점수 > 0: {anomaly_but_high_score:,}개 "
                                f"({anomaly_but_high_score/len(anomaly_scores_transformed)*100:.1f}%)")
                
                # 경계 근처 데이터 분석
                boundary_range = 2  # 결정 경계 ±2
                near_boundary = np.sum(np.abs(scores_transformed - decision_boundary) < boundary_range)
                self.log(f"    - 결정 경계({decision_boundary:.2f}) ±{boundary_range} 범위: "
                        f"{near_boundary:,}개 ({near_boundary/len(scores_transformed)*100:.1f}%)")
                
                # 2차 로직 경계값 추천
                self.log("\n  💡 2차 로직 경계값 추천:")
                
                # 방법 1: 정상 데이터의 하위 퍼센타일
                normal_lower_bound = np.percentile(normal_scores_transformed, 1)  # 정상의 하위 1%
                self.log(f"    - 정상 데이터 하위 1%: {normal_lower_bound:.2f}")
                
                # 방법 2: 전체 데이터의 특정 퍼센타일
                overall_p3 = np.percentile(scores_transformed, 3)
                self.log(f"    - 전체 데이터 하위 3%: {overall_p3:.2f}")
                
                # 방법 3: 평균 - n*표준편차
                mean_minus_2std = np.mean(scores_transformed) - 2 * np.std(scores_transformed)
                mean_minus_3std = np.mean(scores_transformed) - 3 * np.std(scores_transformed)
                self.log(f"    - 평균 - 2σ: {mean_minus_2std:.2f}")
                self.log(f"    - 평균 - 3σ: {mean_minus_3std:.2f}")
                
                # 방법 4: 이상 데이터의 상위 경계
                if len(anomaly_scores_transformed) > 0:
                    anomaly_upper = np.percentile(anomaly_scores_transformed, 90)  # 이상의 상위 10%
                    self.log(f"    - 이상 데이터 상위 10%: {anomaly_upper:.2f}")
                
                # 모델 정보
                model_info = {
                    'machine_id': machine_id,
                    'sensor': sensor,
                    'model_type': 'IsolationForest',
                    'train_samples': len(X_train),
                    'training_periods': self.training_periods,
                    'features': self.sensor_config[sensor]['features'],
                    'best_params': self.study.best_params,
                    'decision_boundary': float(decision_boundary),
                    'boundary_method': 'percentile_5',
                    'score_transform': transform_info,
                    'boundary_stats': {
                        'percentile_5': float(decision_boundary),
                        'method': 'percentile_based'
                    },
                    'anomaly_ratio': float(anomaly_ratio),
                    'score_statistics': {
                        'mean': float(np.mean(scores_transformed)),
                        'std': float(np.std(scores_transformed)),
                        'min': float(np.min(scores_transformed)),
                        'max': float(np.max(scores_transformed))
                    },
                    'normal_score_statistics': {
                        'count': int(len(normal_scores_transformed)),
                        'mean': float(np.mean(normal_scores_transformed)),
                        'std': float(np.std(normal_scores_transformed)),
                        'min': float(np.min(normal_scores_transformed)),
                        'max': float(np.max(normal_scores_transformed)),
                        'percentiles': {
                            'p1': float(np.percentile(normal_scores_transformed, 1)),
                            'p5': float(np.percentile(normal_scores_transformed, 5)),
                            'p10': float(np.percentile(normal_scores_transformed, 10))
                        }
                    },
                    'anomaly_score_statistics': {
                        'count': int(len(anomaly_scores_transformed)),
                        'mean': float(np.mean(anomaly_scores_transformed)) if len(anomaly_scores_transformed) > 0 else None,
                        'std': float(np.std(anomaly_scores_transformed)) if len(anomaly_scores_transformed) > 0 else None,
                        'min': float(np.min(anomaly_scores_transformed)) if len(anomaly_scores_transformed) > 0 else None,
                        'max': float(np.max(anomaly_scores_transformed)) if len(anomaly_scores_transformed) > 0 else None
                    },
                    'score_percentiles': percentile_values,
                    'score_distribution': score_distribution,
                    'secondary_thresholds': {
                        'normal_p1': float(normal_lower_bound),
                        'overall_p3': float(overall_p3),
                        'mean_minus_2std': float(mean_minus_2std),
                        'mean_minus_3std': float(mean_minus_3std),
                        'anomaly_p90': float(anomaly_upper) if len(anomaly_scores_transformed) > 0 else None
                    },
                    'evaluation_skipped': False,
                    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # 모델 저장
            self.log("\n모델 저장 중...")
            timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
            
            # 디렉토리 생성
            model_dir = f"./models/{machine_id}/{sensor}/current_model"
            scale_dir = f"./models/{machine_id}/{sensor}/current_scale"
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(scale_dir, exist_ok=True)
            
            # 기존 파일 삭제
            for d in [model_dir, scale_dir]:
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
            
            # 새 파일 저장
            model_path = os.path.join(model_dir, f"{timestamp}_model.pkl")
            scaler_path = os.path.join(scale_dir, f"{timestamp}_scaler.pkl")
            info_path = os.path.join(model_dir, f"{timestamp}_model_info.json")
            
            joblib.dump(model, model_path)
            scaler.save(scaler_path)
            
            # 🔍 디버깅: 저장된 파일 검증
            self.log(f"\n🔍 [디버깅] 저장된 파일 검증:")
            
            # 모델 재로드 테스트
            test_model = joblib.load(model_path)
            self.log(f"  - 모델 재로드 성공: {type(test_model).__name__}")
            
            # 스케일러 재로드 테스트
            test_scaler = CustomRobustScaler()
            test_scaler.load(scaler_path)
            self.log(f"  - 스케일러 재로드 성공: {len(test_scaler.params)}개 특징")
            
            # 테스트 데이터로 검증
            test_data = X_train[:10]  # 처음 10개 샘플
            test_scaled = test_scaler.transform(test_data)
            test_scores = test_model.score_samples(test_scaled)
            
            # 점수 변환 테스트
            test_transformed, _ = self.transform_scores(test_scores)
            
            self.log(f"  - 테스트 변환: 원본 [{test_data.min():.2f}, {test_data.max():.2f}] → "
                    f"스케일 [{test_scaled.min():.2f}, {test_scaled.max():.2f}]")
            self.log(f"  - 테스트 스코어: 원본 [{test_scores.min():.4f}, {test_scores.max():.4f}] → "
                    f"변환 [{test_transformed.min():.2f}, {test_transformed.max():.2f}]")
            self.log(f"  - 테스트 예측: {test_model.predict(test_scaled)}")
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.log(f"\n✅ 모델 저장 완료:")
            self.log(f"  - 모델: {model_path}")
            self.log(f"  - 스케일러: {scaler_path}")
            self.log(f"  - 정보: {info_path}")
            
            # 현재 모델 정보 저장
            self.current_model_info = model_info
            
            # 전체 소요 시간
            total_time = (datetime.now() - total_start_time).total_seconds()
            self.log(f"\n전체 학습 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
            
            self.progress_var.set("학습 완료!")
            messagebox.showinfo("완료", f"모델 학습이 완료되었습니다.\n머신: {machine_id}\n센서: {sensor}")
            
        except Exception as e:
            self.log(f"❌ 학습 실패: {e}")
            self.progress_var.set("학습 실패")
            messagebox.showerror("오류", f"학습 중 오류 발생: {e}")
        finally:
            self.train_button.config(state='normal')
    
    def start_training(self):
        if not self.training_periods:
            messagebox.showerror("오류", "학습 기간을 추가해주세요.")
            return
        
        if not self.conn:
            messagebox.showerror("오류", "DB 연결이 필요합니다.")
            return
        
        self.train_button.config(state='disabled')
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    
    def test_model(self):
        """모델 테스트"""
        try:
            # 모델 및 스케일러 파일 확인
            model_path = self.model_file_var.get()
            scaler_path = self.scaler_file_var.get()
            
            if not model_path or not scaler_path:
                messagebox.showerror("오류", "모델과 스케일러 파일을 선택해주세요.")
                return
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                messagebox.showerror("오류", "선택한 파일이 존재하지 않습니다.")
                return
            
            # 모델 및 스케일러 로드
            self.log("모델 및 스케일러 로드 중...")
            model = joblib.load(model_path)
            
            scaler = CustomRobustScaler()
            scaler.load(scaler_path)
            
            # 모델 정보 로드
            info_path = model_path.replace('_model.pkl', '_model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
            else:
                # 기본 정보 사용
                model_info = {
                    'decision_boundary': -5.0,
                    'sensor': self.test_sensor_var.get(),
                    'score_transform': None
                }
            
            # 테스트 설정
            machine_id = self.test_machine_var.get()
            sensor = self.test_sensor_var.get()
            
            # 날짜 범위 확인
            start_date = self.test_start_date.get_date()
            end_date = self.test_end_date.get_date()
            
            # 최대 5일 제한
            if (end_date - start_date).days > 5:
                messagebox.showerror("오류", "테스트 기간은 최대 5일까지만 가능합니다.")
                return
            
            if start_date > end_date:
                messagebox.showerror("오류", "시작일이 종료일보다 늦을 수 없습니다.")
                return
            
            # 센서별 데이터베이스 테이블 선택
            table_name = f"normal_{sensor}_data"
            
            # 결과 텍스트 초기화
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Model Test Results\n")
            self.result_text.insert(tk.END, f"{'='*60}\n")
            self.result_text.insert(tk.END, f"Machine: {machine_id}, Sensor: {sensor}\n")
            self.result_text.insert(tk.END, f"Table: {table_name}\n")
            self.result_text.insert(tk.END, f"Period: {start_date} ~ {end_date}\n")
            self.result_text.insert(tk.END, f"Sampling: {self.sampling_info_var.get()}\n")
            self.result_text.insert(tk.END, f"Decision Boundary: {model_info.get('decision_boundary', 'N/A')}\n\n")
            
            # 전체 기간 데이터를 한 번에 가져오기
            self.log(f"Extracting test data: {start_date} ~ {end_date}")
            
            # 전체 기간 데이터 추출 (한 번에)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # get_training_data 함수 사용 (전체 기간)
            test_features, test_timestamps = self.get_training_data(machine_id, sensor, start_str, end_str, "00:00", "23:59", return_timestamps=True)
            
            if test_features is None or len(test_features) == 0:
                self.result_text.insert(tk.END, "No data found for the selected period.\n")
                self.log("No test data found")
                return
            
            self.result_text.insert(tk.END, f"Processing {len(test_features)} windows...\n")
            self.root.update_idletasks()
            
            # 스케일링
            X_test_scaled = scaler.transform(test_features)
            self.log(f"✅ Test data normalized: {X_test_scaled.shape}")
            
            # 예측
            self.log("Running model predictions...")
            predictions = model.predict(X_test_scaled)
            scores = model.score_samples(X_test_scaled)
            
            # 점수 변환
            if model_info.get('score_transform'):
                scores_transformed, _ = self.transform_scores(scores)
            else:
                min_score = np.min(scores)
                max_score = np.max(scores)
                scores_transformed = -10 + (scores - min_score) * 20 / (max_score - min_score)
            
            self.log(f"✅ Prediction complete:")
            self.log(f"  - Original score range: [{scores.min():.4f}, {scores.max():.4f}]")
            self.log(f"  - Transformed score range: [{scores_transformed.min():.2f}, {scores_transformed.max():.2f}]")
            
            # 실제 타임스탬프 사용
            all_timestamps = pd.to_datetime(test_timestamps)
            all_scores = scores_transformed
            
            # 플롯 그리기
            self.plot_test_results(all_timestamps, all_scores, sensor, model_info)
            
            # 전체 결과 요약
            if len(all_scores) > 0:
                all_scores = np.array(all_scores)
                decision_boundary_value = float(model_info.get('decision_boundary', 0))
                total_anomalies = np.sum(all_scores < decision_boundary_value)
                total_ratio = total_anomalies / len(all_scores) * 100
                
                self.result_text.insert(tk.END, f"\n{'='*60}\n")
                self.result_text.insert(tk.END, f"Overall Summary\n")
                self.result_text.insert(tk.END, f"Total windows: {len(all_scores):,}\n")
                self.result_text.insert(tk.END, f"Anomalies detected: {total_anomalies:,} ({total_ratio:.2f}%)\n")
                self.result_text.insert(tk.END, f"Score range: [{np.min(all_scores):.2f}, {np.max(all_scores):.2f}]\n")
                self.result_text.insert(tk.END, f"Average score: {np.mean(all_scores):.2f} ± {np.std(all_scores):.2f}\n")
                
                # 날짜별 통계 추가
                df_results = pd.DataFrame({'timestamp': all_timestamps, 'score': all_scores})
                df_results['date'] = df_results['timestamp'].dt.date
                daily_stats = df_results.groupby('date')['score'].agg(['count', 'mean', 'std'])
                
                self.result_text.insert(tk.END, f"\nDaily Statistics:\n")
                for date, stats in daily_stats.iterrows():
                    # 날짜별 데이터 필터링
                    date_data = df_results[df_results['date'] == date]
                    date_scores = date_data['score'].values
                    anomaly_count = np.sum(date_scores < decision_boundary_value)
                    self.result_text.insert(tk.END, f"  {date}: {stats['count']} windows, {anomaly_count} anomalies, avg score: {stats['mean']:.2f}\n")
            
        except Exception as e:
            messagebox.showerror("오류", f"테스트 중 오류 발생: {e}")
            self.log(f"테스트 오류: {e}")
    
    def plot_test_results(self, timestamps, scores, sensor, model_info):
        """테스트 결과를 시계열 플롯으로 표시"""
        try:
            # 기존 플롯 클리어
            self.ax1.clear()
            self.ax2.clear()
            
            # numpy 배열로 변환
            scores = np.array(scores)
            timestamps = pd.to_datetime(timestamps)
            decision_boundary = float(model_info.get('decision_boundary', 0))  # float으로 변환
            
            # 데이터가 없는 경우 처리
            if len(scores) == 0:
                self.ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=self.ax1.transAxes)
                self.ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=self.ax2.transAxes)
                return
            
            # 1. Score 플롯 (개선된 X축)
            self.ax1.plot(timestamps, scores, 'b-', linewidth=0.5, alpha=0.7, label='Score')
            self.ax1.axhline(y=decision_boundary, color='r', linestyle='--', linewidth=2, 
                            label=f'Decision Boundary ({decision_boundary:.2f})')
            self.ax1.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
            
            # 이상 구간 표시
            anomaly_mask = scores < decision_boundary
            if np.any(anomaly_mask):
                self.ax1.scatter(timestamps[anomaly_mask],
                               scores[anomaly_mask],
                               color='red', s=10, alpha=0.5, label='Anomaly')
            
            self.ax1.set_ylabel('Score', fontsize=12)
            self.ax1.set_title(f'{self.test_machine_var.get()} - {sensor.upper()} Anomaly Detection Results', fontsize=14)
            self.ax1.legend(loc='upper right')
            self.ax1.grid(True, alpha=0.3)
            
            # X축 포맷 개선
            # 날짜 범위에 따라 적절한 포맷 선택
            date_range = (timestamps[-1] - timestamps[0]).days
            
            if date_range == 0:  # 같은 날
                # 시간 범위 확인
                time_range_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
                
                if time_range_hours <= 6:  # 6시간 이하
                    self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    self.ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                    self.ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
                elif time_range_hours <= 12:  # 12시간 이하
                    self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    self.ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    self.ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
                else:  # 하루 전체
                    self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    self.ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                    self.ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            elif date_range <= 3:  # 3일 이하
                self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                self.ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            elif date_range <= 7:  # 1주일 이하
                self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                self.ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            
            plt.setp(self.ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 2. 날짜별 이상치 비율 막대 그래프
            # DataFrame 생성
            df = pd.DataFrame({
                'timestamp': timestamps,
                'score': scores,
                'is_anomaly': anomaly_mask
            })
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            
            # 날짜별 이상치 비율 계산
            daily_stats = df.groupby('date').agg({
                'is_anomaly': ['sum', 'count']
            })
            daily_stats.columns = ['anomaly_count', 'total_count']
            daily_stats['anomaly_ratio'] = daily_stats['anomaly_count'] / daily_stats['total_count'] * 100
            
            # 막대 그래프
            dates = daily_stats.index
            ratios = daily_stats['anomaly_ratio'].values
            
            # dates를 matplotlib이 이해할 수 있는 형식으로 변환
            if len(dates) > 0:
                # pandas date를 datetime으로 변환
                dates_for_plot = [pd.to_datetime(d) for d in dates]
                bars = self.ax2.bar(dates_for_plot, ratios, alpha=0.7, color='red')
            else:
                bars = []
            
            # 데이터가 없는 날짜 처리
            if len(bars) == 0:
                self.ax2.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=self.ax2.transAxes)
                self.fig.tight_layout()
                self.canvas.draw()
                return
            
            # 값 레이블 추가 (막대 위에)
            for bar, ratio in zip(bars, ratios):
                if ratio > 0:
                    height = bar.get_height()
                    self.ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{ratio:.1f}%',
                                ha='center', va='bottom', fontsize=8)
            
            self.ax2.set_ylabel('Daily Anomaly Ratio (%)', fontsize=12)
            self.ax2.set_xlabel('Date', fontsize=12)
            self.ax2.grid(True, alpha=0.3, axis='y')
            
            # X축 포맷
            if len(dates) <= 7:
                self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                self.ax2.xaxis.set_major_locator(mdates.DayLocator())
            else:
                self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                self.ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
            
            plt.setp(self.ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 평균선 추가
            mean_ratio = daily_stats['anomaly_ratio'].mean()
            self.ax2.axhline(y=mean_ratio, color='orange', linestyle=':', linewidth=2, 
                            label=f'Average: {mean_ratio:.1f}%')
            self.ax2.legend()
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # 추가 정보를 결과 텍스트에 출력
            self.result_text.insert(tk.END, f"\n\n📊 Hourly Anomaly Analysis:\n")
            
            # 시간대별 통계
            hourly_stats = df.groupby('hour')['is_anomaly'].agg(['sum', 'count'])
            hourly_stats['ratio'] = hourly_stats['sum'] / hourly_stats['count'] * 100
            
            self.result_text.insert(tk.END, f"\nAverage anomaly ratio by hour:\n")
            for hour in range(24):
                if hour in hourly_stats.index:
                    ratio = hourly_stats.loc[hour, 'ratio']
                    count = hourly_stats.loc[hour, 'count']
                    if ratio > 0:
                        self.result_text.insert(tk.END, f"  {hour:02d}:00: {ratio:5.1f}% (n={count})\n")
            
            # 실제 데이터 간격 정보 추가
            if len(timestamps) > 1:
                time_diffs = np.diff(timestamps.values).astype('timedelta64[s]').astype(float)
                avg_interval = np.mean(time_diffs)
                std_interval = np.std(time_diffs)
                
                self.result_text.insert(tk.END, f"\n📊 Data Collection Interval:\n")
                self.result_text.insert(tk.END, f"  Average: {avg_interval:.1f} seconds\n")
                self.result_text.insert(tk.END, f"  Std Dev: {std_interval:.1f} seconds\n")
                self.result_text.insert(tk.END, f"  Range: [{np.min(time_diffs):.1f}, {np.max(time_diffs):.1f}] seconds\n")
            
        except Exception as e:
            self.log(f"플롯 생성 오류: {e}")
    
    def analyze_hourly_anomalies(self, machine_id, sensor, test_date, predictions, scores):
        """시간대별 이상 탐지 분석"""
        try:
            # 해당 날짜의 원본 데이터 다시 로드 (시간 정보 포함)
            if sensor == 'acc':
                query = """
                SELECT time, x, y, z
                FROM normal_acc_data
                WHERE machine_id = %s
                AND DATE(time) = %s
                ORDER BY time
                """
            else:
                query = """
                SELECT time, mic_value
                FROM normal_mic_data
                WHERE machine_id = %s
                AND DATE(time) = %s
                ORDER BY time
                """
            
            df = pd.read_sql(query, self.conn, params=(machine_id, test_date))
            df['time'] = pd.to_datetime(df['time'])
            
            # 5초 윈도우로 그룹화 (학습과 동일)
            window_sec = self.sensor_config[sensor]['window_sec']
            df['window'] = df['time'].dt.floor(f'{window_sec}S')
            df['hour'] = df['time'].dt.hour
            
            # 각 윈도우의 시간대 할당
            window_hours = df.groupby('window')['hour'].first().values
            
            # 최소 데이터 요구사항을 만족하는 윈도우만 필터링
            window_samples = window_sec * 10  # DB는 10Hz
            valid_windows = df.groupby('window').size() >= window_samples * 0.8
            valid_indices = valid_windows[valid_windows].index
            
            # 유효한 윈도우의 인덱스 찾기
            window_idx = 0
            hourly_predictions = {h: [] for h in range(24)}
            hourly_scores = {h: [] for h in range(24)}
            
            for window in valid_indices:
                if window_idx < len(predictions):
                    hour = df[df['window'] == window]['hour'].iloc[0]
                    hourly_predictions[hour].append(predictions[window_idx])
                    hourly_scores[hour].append(scores[window_idx])
                    window_idx += 1
            
            # 시간대별 통계 계산
            hourly_stats = {}
            for hour in range(24):
                if len(hourly_predictions[hour]) > 0:
                    anomaly_count = sum(p == -1 for p in hourly_predictions[hour])
                    total_count = len(hourly_predictions[hour])
                    
                    hourly_stats[hour] = {
                        'anomaly_count': anomaly_count,
                        'total_count': total_count,
                        'anomaly_ratio': anomaly_count / total_count * 100,
                        'mean_score': np.mean(hourly_scores[hour]),
                        'min_score': np.min(hourly_scores[hour]),
                        'max_score': np.max(hourly_scores[hour])
                    }
            
            return hourly_stats
            
        except Exception as e:
            self.log(f"시간대별 분석 오류: {e}")
            return None
    
    def transform_scores(self, scores):
        """Isolation Forest 점수를 자연스러운 이상 점수로 변환"""
        # Isolation Forest 점수 특성:
        # - 0에 가까울수록 정상
        # - 음수일수록 이상
        # - 일반적으로 -0.1 ~ -0.7 범위
        
        # 중앙값을 기준점으로 사용
        median_score = np.median(scores)
        
        # 비선형 변환 함수 (지수 함수 기반)
        def exponential_transform(x):
            # 중앙값 기준으로 정규화
            normalized = (x - median_score) / abs(median_score) if median_score != 0 else x
            
            # 지수 변환으로 자연스러운 분포 생성
            if normalized >= 0:
                # 정상 범위: 0 근처 유지
                return normalized * 2  # 최대 +2 정도
            else:
                # 이상 범위: 지수적으로 증가
                # normalized가 -1일 때 약 -7
                # normalized가 -2일 때 약 -15
                # normalized가 -3일 때 약 -25
                return 5 * (np.exp(abs(normalized)) - 1) * np.sign(normalized)
        
        # 벡터화된 변환 적용
        transformed = np.vectorize(exponential_transform)(scores)
        
        # 클리핑 없음 - 자연스러운 값 그대로 사용
        
        # 통계 정보
        stats = {
            'median_score': float(median_score),
            'original_range': [float(np.min(scores)), float(np.max(scores))],
            'transformed_range': [float(np.min(transformed)), float(np.max(transformed))]
        }
        
        return transformed, stats
    
    def start_testing(self):
        start_date = self.test_start_date.get_date()
        end_date = self.test_end_date.get_date()
        
        if start_date > end_date:
            messagebox.showerror("오류", "시작일이 종료일보다 늦을 수 없습니다.")
            return
        
        thread = threading.Thread(target=self.test_model)
        thread.daemon = True
        thread.start()

def main():
    root = tk.Tk()
    app = OCSVMTrainerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()