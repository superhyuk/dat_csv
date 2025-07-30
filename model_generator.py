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
from sklearn.svm import OneClassSVM
import optuna
from optuna import create_study
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
                "nu_range": [0.01, 0.15],
                "gamma_range": [0.0001, 0.005]
            },
            "acc": {
                "sampling_rate": 1666,
                "window_sec": 5,
                "features": ["x_peak", "x_crest_factor", "y_peak", "y_crest_factor", "z_peak", "z_crest_factor"],
                "nu_range": [0.01, 0.15],
                "gamma_range": [0.0001, 0.005]
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
        self.trials_var = tk.IntVar(value=50)
        trials_spinbox = ttk.Spinbox(config_frame, from_=10, to=500, increment=10,
                                    textvariable=self.trials_var, width=10)
        trials_spinbox.grid(row=row, column=1, padx=5)
        
        # 학습 기간 프레임
        period_frame = ttk.LabelFrame(parent, text="학습 기간 설정", padding="10")
        period_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 기간 입력
        input_frame = ttk.Frame(period_frame)
        input_frame.pack(fill='x')
        
        ttk.Label(input_frame, text="시작일:").pack(side='left', padx=5)
        self.train_start_date = ttk.Entry(input_frame, width=12)
        self.train_start_date.pack(side='left', padx=5)
        
        ttk.Label(input_frame, text="종료일:").pack(side='left', padx=5)
        self.train_end_date = ttk.Entry(input_frame, width=12)
        self.train_end_date.pack(side='left', padx=5)
        
        ttk.Button(input_frame, text="기간 추가", 
                  command=self.add_training_period).pack(side='left', padx=20)
        
        # 기간 목록
        list_frame = ttk.Frame(period_frame)
        list_frame.pack(fill='both', expand=True, pady=10)
        
        # 트리뷰
        columns = ('시작일', '종료일', '일수')
        self.train_tree = ttk.Treeview(list_frame, columns=columns, height=8)
        self.train_tree.heading('#0', text='No')
        self.train_tree.heading('시작일', text='시작일')
        self.train_tree.heading('종료일', text='종료일')
        self.train_tree.heading('일수', text='일수')
        
        self.train_tree.column('#0', width=50)
        self.train_tree.column('시작일', width=150)
        self.train_tree.column('종료일', width=150)
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
        
        ttk.Label(test_input_frame, text="날짜:").pack(side='left', padx=5)
        self.test_date = ttk.Entry(test_input_frame, width=12)
        self.test_date.pack(side='left', padx=5)
        
        ttk.Button(test_input_frame, text="날짜 추가", 
                  command=self.add_test_date).pack(side='left', padx=20)
        
        # 테스트 날짜 목록
        self.test_listbox = tk.Listbox(test_period_frame, height=8)
        self.test_listbox.pack(fill='both', expand=True, pady=10)
        
        # 버튼
        test_button_frame = ttk.Frame(test_period_frame)
        test_button_frame.pack(fill='x')
        
        ttk.Button(test_button_frame, text="선택 삭제", 
                  command=self.remove_test_date).pack(side='left', padx=5)
        ttk.Button(test_button_frame, text="모두 삭제", 
                  command=self.clear_test_dates).pack(side='left', padx=5)
        
        # 테스트 시작 버튼
        ttk.Button(parent, text="테스트 시작", 
                  command=self.start_testing).pack(pady=10)
        
        # 결과 표시
        result_frame = ttk.LabelFrame(parent, text="테스트 결과", padding="10")
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
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
                self.train_start_date.insert(0, min_date.strftime("%Y-%m-%d"))
                self.train_end_date.insert(0, max_date.strftime("%Y-%m-%d"))
                self.test_date.insert(0, max_date.strftime("%Y-%m-%d"))
                
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
    
    def add_training_period(self):
        try:
            start = self.train_start_date.get()
            end = self.train_end_date.get()
            
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
            
            if start_date >= end_date:
                messagebox.showerror("오류", "시작일이 종료일보다 늦습니다.")
                return
            
            days = (end_date - start_date).days + 1
            
            # 트리에 추가
            item_id = self.train_tree.insert('', 'end', 
                                           text=str(len(self.training_periods) + 1),
                                           values=(start, end, days))
            
            self.training_periods.append((start, end))
            self.log(f"학습 기간 추가: {start} ~ {end} ({days}일)")
            
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
    
    def add_test_date(self):
        date = self.test_date.get()
        try:
            datetime.strptime(date, "%Y-%m-%d")
            if date not in self.test_listbox.get(0, tk.END):
                self.test_listbox.insert(tk.END, date)
                self.test_periods.append(date)
        except ValueError:
            messagebox.showerror("오류", "날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)")
    
    def remove_test_date(self):
        selected = self.test_listbox.curselection()
        for idx in reversed(selected):
            self.test_listbox.delete(idx)
            del self.test_periods[idx]
    
    def clear_test_dates(self):
        self.test_listbox.delete(0, tk.END)
        self.test_periods.clear()
    
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
    
    def get_training_data(self, machine_id, sensor, start_date, end_date):
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
            AND time >= %s AND time <= %s
            ORDER BY time
            """
        else:  # mic
            query = """
            SELECT time, mic_value
            FROM normal_mic_data
            WHERE machine_id = %s
            AND time >= %s AND time <= %s
            ORDER BY time
            """
        
        self.log(f"데이터 추출 중: {machine_id}, {sensor}, {start_date} ~ {end_date}")
        
        try:
            df = pd.read_sql(query, self.conn, 
                            params=(machine_id, start_date, end_date))
            
            if df.empty:
                self.log(f"데이터가 없습니다!")
                return None
            
            self.log(f"전체 데이터: {len(df)}개 샘플")
            
            # Python에서 5초 윈도우로 분할
            features_list = []
            
            # 시간을 datetime으로 변환
            df['time'] = pd.to_datetime(df['time'])
            
            # 5초 단위로 그룹화
            df['window'] = df['time'].dt.floor(f'{window_sec}S')
            
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
                        
                        # 진행 상황 로그 (1000개마다)
                        if window_count % 1000 == 0:
                            self.log(f"  처리된 윈도우: {window_count}개")
                            
                    except Exception as e:
                        self.log(f"  윈도우 처리 오류: {e}")
                        continue
            
            self.log(f"추출된 윈도우: {len(features_list)}개")
            
            return np.array(features_list) if features_list else None
            
        except Exception as e:
            self.log(f"데이터 추출 오류: {e}")
            return None
    
    def train_model(self):
        """모델 학습 (별도 스레드에서 실행)"""
        try:
            machine_id = self.machine_var.get()
            sensor = self.sensor_var.get()
            n_trials = self.trials_var.get()
            
            self.log(f"\n{'='*60}")
            self.log(f"{machine_id} / {sensor} 모델 학습 시작")
            self.log(f"{'='*60}")
            
            # 학습 데이터 수집
            all_features = []
            period_info = []  # 각 기간별 정보 저장
            
            for start, end in self.training_periods:
                self.progress_var.set(f"데이터 추출 중: {start} ~ {end}")
                features = self.get_training_data(machine_id, sensor, start, end)
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    period_info.append({
                        'period': f"{start} ~ {end}",
                        'start_idx': len(np.vstack(all_features[:-1])) if len(all_features) > 1 else 0,
                        'end_idx': len(np.vstack(all_features)),
                        'count': len(features)
                    })
                    self.log(f"✅ {start} ~ {end}: {len(features)}개 윈도우")
            
            if not all_features:
                self.log("❌ 학습 데이터가 없습니다.")
                self.progress_var.set("학습 데이터 없음")
                return
            
            X_train = np.vstack(all_features)
            self.log(f"전체 학습 데이터: {X_train.shape}")
            
            # 스케일러 학습
            self.progress_var.set("스케일러 학습 중...")
            scaler = CustomRobustScaler()
            X_scaled = scaler.fit_transform(X_train)
            self.log("✅ 스케일러 학습 완료")
            
            # OCSVM 최적화
            self.progress_var.set(f"하이퍼파라미터 최적화 중... (0/{n_trials})")
            
            opt_config = self.sensor_config[sensor]
            nu_range = opt_config['nu_range']
            gamma_range = opt_config['gamma_range']
            
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
                X_sample = X_scaled[sample_indices]
                self.log(f"✅ 총 {len(X_sample)}개 샘플 추출 완료")
            else:
                X_sample = X_scaled
                self.log(f"📊 데이터가 충분히 작아 전체 사용: {len(X_scaled)}개")
            
            # study 변수를 클래스 변수로 만들어 objective 함수에서 접근 가능하게
            self.study = create_study(direction='maximize')
            
            def objective(trial):
                # 진행률 업데이트
                current_trial = len(self.study.trials)
                self.progress_var.set(f"하이퍼파라미터 최적화 중... ({current_trial}/{n_trials})")
                
                nu = trial.suggest_float('nu', nu_range[0], nu_range[1], log=True)
                gamma = trial.suggest_float('gamma', gamma_range[0], gamma_range[1], log=True)
                
                model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                model.fit(X_sample)
                
                scores = model.decision_function(X_sample)
                threshold = np.percentile(scores, 5)
                predictions = (scores > threshold).astype(int)
                accuracy = np.mean(predictions)
                
                return accuracy
            
            self.study.optimize(objective, n_trials=n_trials)
            
            best_params = self.study.best_params
            self.log(f"✅ 최적 파라미터: {best_params}")
            
            # 최종 모델 학습
            self.progress_var.set("최종 모델 학습 중...")
            model = OneClassSVM(
                kernel='rbf',
                nu=best_params['nu'],
                gamma=best_params['gamma']
            )
            model.fit(X_scaled)
            
            # 결정 경계 계산
            scores = model.decision_function(X_scaled)
            decision_boundary = float(np.percentile(scores, 5))
            
            # 모델 정보
            model_info = {
                'machine_id': machine_id,
                'sensor': sensor,
                'train_samples': len(X_train),
                'training_periods': self.training_periods,
                'features': self.sensor_config[sensor]['features'],
                'best_params': best_params,
                'decision_boundary': decision_boundary,
                'score_statistics': {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                },
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 모델 저장
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
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.log(f"\n✅ 모델 저장 완료:")
            self.log(f"  - 모델: {model_path}")
            self.log(f"  - 스케일러: {scaler_path}")
            self.log(f"  - 정보: {info_path}")
            
            # 현재 모델 정보 저장
            self.current_model_info = model_info
            
            self.progress_var.set("학습 완료!")
            messagebox.showinfo("완료", "모델 학습이 완료되었습니다.")
            
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
                    'sensor': self.test_sensor_var.get()
                }
            
            # 테스트 설정
            machine_id = self.test_machine_var.get()
            sensor = self.test_sensor_var.get()
            
            # 센서별 데이터베이스 테이블 선택
            table_name = f"normal_{sensor}_data"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"모델 테스트 결과\n")
            self.result_text.insert(tk.END, f"{'='*60}\n")
            self.result_text.insert(tk.END, f"머신: {machine_id}, 센서: {sensor}\n")
            self.result_text.insert(tk.END, f"테이블: {table_name}\n")
            self.result_text.insert(tk.END, f"샘플링: {self.sampling_info_var.get()}\n")
            self.result_text.insert(tk.END, f"결정 경계: {model_info.get('decision_boundary', 'N/A')}\n\n")
            
            for test_date in self.test_periods:
                self.result_text.insert(tk.END, f"\n[{test_date}]\n")
                
                # 테스트 데이터 추출 (라즈베리파이와 동일한 방식)
                test_data = self.get_training_data(
                    machine_id, sensor,
                    f"{test_date} 00:00:00",
                    f"{test_date} 23:59:59"
                )
                
                if test_data is None or len(test_data) == 0:
                    self.result_text.insert(tk.END, "  - 데이터 없음\n")
                    continue
                
                # 예측
                X_test_scaled = scaler.transform(test_data)
                predictions = model.predict(X_test_scaled)
                scores = model.decision_function(X_test_scaled)
                
                # 결과 분석
                anomaly_count = np.sum(predictions == -1)
                anomaly_ratio = anomaly_count / len(predictions) * 100
                
                self.result_text.insert(tk.END, f"  - 샘플 수: {len(test_data)}\n")
                self.result_text.insert(tk.END, f"  - 이상 탐지: {anomaly_count}개 ({anomaly_ratio:.1f}%)\n")
                self.result_text.insert(tk.END, f"  - 점수: 평균={np.mean(scores):.3f}, ")
                self.result_text.insert(tk.END, f"최소={np.min(scores):.3f}, 최대={np.max(scores):.3f}\n")
                
                # 점수 분포
                self.result_text.insert(tk.END, f"  - 점수 < 0: {np.sum(scores < 0)}개\n")
                self.result_text.insert(tk.END, f"  - 점수 < -5: {np.sum(scores < -5)}개\n")
                self.result_text.insert(tk.END, f"  - 점수 < -10: {np.sum(scores < -10)}개\n")
                
                # 결정 경계 기준 이상 탐지
                if 'decision_boundary' in model_info:
                    boundary = model_info['decision_boundary']
                    below_boundary = np.sum(scores < boundary)
                    self.result_text.insert(tk.END, f"  - 점수 < {boundary:.3f} (결정경계): {below_boundary}개\n")
                
            self.result_text.insert(tk.END, f"\n{'='*60}\n")
            self.result_text.insert(tk.END, "테스트 완료\n")
            
        except Exception as e:
            messagebox.showerror("오류", f"테스트 중 오류 발생: {e}")
    
    def start_testing(self):
        if not self.test_periods:
            messagebox.showerror("오류", "테스트 날짜를 추가해주세요.")
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