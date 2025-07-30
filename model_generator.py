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
        self.trials_var = tk.IntVar(value=50)
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
            start_time = datetime.now()
            df = pd.read_sql(query, self.conn, 
                            params=(machine_id, start_date, end_date))
            
            if df.empty:
                self.log(f"데이터가 없습니다!")
                return None
            
            self.log(f"전체 데이터: {len(df)}개 샘플")
            self.log(f"데이터 로드 시간: {(datetime.now() - start_time).total_seconds():.1f}초")
            
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
                            elapsed = (datetime.now() - start_time).total_seconds()
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
            
            total_time = (datetime.now() - start_time).total_seconds()
            self.log(f"추출 완료: {len(features_list)}개 윈도우 (총 {total_time:.1f}초)")
            
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
            
            for idx, (start, end) in enumerate(self.training_periods):
                self.log(f"\n[{idx+1}/{len(self.training_periods)}] 기간: {start} ~ {end}")
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
                    self.log(f"✅ 추출 성공: {len(features)}개 윈도우")
            
            if not all_features:
                self.log("❌ 학습 데이터가 없습니다.")
                self.progress_var.set("학습 데이터 없음")
                return
            
            X_train = np.vstack(all_features)
            self.log(f"\n전체 학습 데이터: {X_train.shape}")
            
            # 스케일러 학습
            self.log("\n스케일러 학습 시작...")
            self.progress_var.set("스케일러 학습 중...")
            scaler_start = datetime.now()
            scaler = CustomRobustScaler()
            X_scaled = scaler.fit_transform(X_train)
            scaler_time = (datetime.now() - scaler_start).total_seconds()
            self.log(f"✅ 스케일러 학습 완료 ({scaler_time:.1f}초)")
            
            # OCSVM 최적화
            self.log(f"\n하이퍼파라미터 최적화 시작 (Optuna {n_trials} trials)")
            optuna_start = datetime.now()
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
            
            trial_count = 0
            # study 변수를 클래스 변수로 만들어 objective 함수에서 접근 가능하게
            self.study = create_study(direction='maximize')
            
            def objective(trial):
                # 진행률 업데이트
                current_trial = len(self.study.trials)
                self.progress_var.set(f"하이퍼파라미터 최적화 중... ({current_trial}/{n_trials})")
                self.root.update_idletasks()
                
                nu = trial.suggest_float('nu', nu_range[0], nu_range[1], log=True)
                gamma = trial.suggest_float('gamma', gamma_range[0], gamma_range[1], log=True)
                
                model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                model.fit(X_sample)
                
                scores = model.decision_function(X_sample)
                threshold = np.percentile(scores, 5)
                predictions = (scores > threshold).astype(int)
                score = np.mean(predictions)
                
                return score
            
            # Optuna 콜백 함수
            def optuna_callback(study, trial):
                nonlocal trial_count
                trial_count += 1
                if trial_count % 10 == 0 or trial_count <= 5:
                    self.log(f"  Trial {trial_count}: nu={trial.params['nu']:.4f}, "
                            f"gamma={trial.params['gamma']:.6f}, score={trial.value:.4f}")
            
            # 최적화 실행
            self.study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
            
            optuna_time = (datetime.now() - optuna_start).total_seconds()
            self.log(f"\n✅ 최적화 완료 ({optuna_time:.1f}초)")
            self.log(f"최적 파라미터: nu={self.study.best_params['nu']:.4f}, "
                    f"gamma={self.study.best_params['gamma']:.6f}")
            self.log(f"최적 점수: {self.study.best_value:.4f}")
            
            # 최적 모델로 전체 데이터 학습
            self.progress_var.set("최종 모델 학습 중...")
            self.log("\n최종 모델 학습 시작...")
            best_nu = self.study.best_params['nu']
            best_gamma = self.study.best_params['gamma']
            
            model = OneClassSVM(kernel='rbf', nu=best_nu, gamma=best_gamma)
            model.fit(X_scaled)
            self.log("✅ 최종 모델 학습 완료")
            
            # 모델 성능 평가 (선택적)
            skip_evaluation = self.skip_eval_var.get()  # GUI 체크박스 값 사용
            
            if skip_evaluation:
                self.log("\n⚡ 성능 평가 단계 스킵 (빠른 학습 모드)")
                # 간단한 샘플링으로 대략적인 성능만 확인
                sample_size = min(10000, len(X_scaled))
                sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
                sample_scores = model.decision_function(X_scaled[sample_indices])
                decision_boundary = np.percentile(sample_scores, 5)
                
                model_info = {
                    'machine_id': machine_id,
                    'sensor': sensor,
                    'train_samples': len(X_train),
                    'training_periods': self.training_periods,
                    'features': self.sensor_config[sensor]['features'],
                    'best_params': self.study.best_params,
                    'decision_boundary': float(decision_boundary),
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
                    batch_scores = model.decision_function(batch)
                    
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
                
                # 성능 지표 계산
                self.log("\n성능 지표 계산 중...")
                anomaly_ratio = np.sum(predictions == -1) / len(predictions) * 100
                decision_boundary = float(np.percentile(scores, 5))
                
                self.log(f"✅ 학습 완료!")
                self.log(f"  - 이상치 비율: {anomaly_ratio:.2f}%")
                self.log(f"  - 결정 경계: {decision_boundary:.4f}")
                self.log(f"  - 점수 범위: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
                self.log(f"  - 점수 평균±표준편차: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
                # 기간별 성능 분석
                self.log("\n📊 기간별 성능:")
                for info in period_info:
                    start_idx = info['start_idx']
                    end_idx = info['end_idx']
                    period_scores = scores[start_idx:end_idx]
                    period_predictions = predictions[start_idx:end_idx]
                    period_anomaly_ratio = np.sum(period_predictions == -1) / len(period_predictions) * 100
                    
                    self.log(f"  - {info['period']}: 이상 {period_anomaly_ratio:.1f}%, "
                            f"점수 {np.mean(period_scores):.2f}±{np.std(period_scores):.2f}")
                
                # 2차 로직을 위한 상세 통계 분석
                self.log("\n📊 2차 로직 경계값 설정을 위한 분석:")
                
                # 정상/이상 데이터 분리
                normal_scores = scores[predictions == 1]
                anomaly_scores = scores[predictions == -1]
                
                # 퍼센타일 기반 경계값 후보
                percentiles = [0.1, 0.5, 1, 2, 3, 5, 10, 15, 20]
                percentile_values = {}
                
                self.log("\n  정상 데이터 점수 분포:")
                self.log(f"    - 개수: {len(normal_scores):,}개 ({len(normal_scores)/len(scores)*100:.1f}%)")
                self.log(f"    - 평균±표준편차: {np.mean(normal_scores):.2f} ± {np.std(normal_scores):.2f}")
                self.log(f"    - 최소/최대: {np.min(normal_scores):.2f} / {np.max(normal_scores):.2f}")
                
                self.log("\n  이상 데이터 점수 분포:")
                self.log(f"    - 개수: {len(anomaly_scores):,}개 ({len(anomaly_scores)/len(scores)*100:.1f}%)")
                self.log(f"    - 평균±표준편차: {np.mean(anomaly_scores):.2f} ± {np.std(anomaly_scores):.2f}")
                self.log(f"    - 최소/최대: {np.min(anomaly_scores):.2f} / {np.max(anomaly_scores):.2f}")
                
                self.log("\n  전체 점수 퍼센타일:")
                for p in percentiles:
                    val = np.percentile(scores, p)
                    percentile_values[f"p{p}"] = float(val)
                    self.log(f"    - {p:5.1f}%: {val:8.2f}")
                
                # 점수 구간별 분포
                self.log("\n  점수 구간별 분포:")
                score_ranges = [
                    (-np.inf, -100, "극심한 이상"),
                    (-100, -50, "심각한 이상"),
                    (-50, -20, "중간 이상"),
                    (-20, -10, "경미한 이상"),
                    (-10, 0, "의심 구간"),
                    (0, 100, "정상 범위"),
                    (100, np.inf, "매우 정상")
                ]
                
                score_distribution = {}
                self.log("    [전체 데이터]")
                for min_score, max_score, label in score_ranges:
                    count = np.sum((scores >= min_score) & (scores < max_score))
                    ratio = count / len(scores) * 100
                    self.log(f"    - {label:12s} [{min_score:6.0f} ~ {max_score:6.0f}]: "
                            f"{count:6,}개 ({ratio:5.1f}%)")
                
                # 정상 데이터의 점수 구간별 분포
                self.log("\n    [정상으로 분류된 데이터]")
                normal_distribution = {}
                for min_score, max_score, label in score_ranges:
                    count = np.sum((normal_scores >= min_score) & (normal_scores < max_score))
                    ratio = count / len(normal_scores) * 100 if len(normal_scores) > 0 else 0
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
                    count = np.sum((anomaly_scores >= min_score) & (anomaly_scores < max_score))
                    ratio = count / len(anomaly_scores) * 100 if len(anomaly_scores) > 0 else 0
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
                    total_count = np.sum((scores >= min_score) & (scores < max_score))
                    normal_count = np.sum((normal_scores >= min_score) & (normal_scores < max_score))
                    anomaly_count = np.sum((anomaly_scores >= min_score) & (anomaly_scores < max_score))
                    
                    score_distribution[label] = {
                        'total': {
                            'count': int(total_count),
                            'ratio': float(total_count / len(scores) * 100)
                        },
                        'normal': {
                            'count': int(normal_count),
                            'ratio': float(normal_count / len(normal_scores) * 100) if len(normal_scores) > 0 else 0,
                            'of_total': float(normal_count / total_count * 100) if total_count > 0 else 0
                        },
                        'anomaly': {
                            'count': int(anomaly_count),
                            'ratio': float(anomaly_count / len(anomaly_scores) * 100) if len(anomaly_scores) > 0 else 0,
                            'of_total': float(anomaly_count / total_count * 100) if total_count > 0 else 0
                        },
                        'range': [float(min_score) if min_score != -np.inf else None,
                                 float(max_score) if max_score != np.inf else None]
                    }
                
                # 교차 분석
                self.log("\n  📊 정상/이상 교차 분석:")
                
                # 정상으로 분류되었지만 점수가 낮은 데이터
                normal_but_low_score = np.sum(normal_scores < 0)
                if normal_but_low_score > 0:
                    self.log(f"    - 정상 분류지만 점수 < 0: {normal_but_low_score:,}개 "
                            f"({normal_but_low_score/len(normal_scores)*100:.1f}%)")
                    
                    # 상세 분포
                    for threshold in [-10, -20, -50, -100]:
                        count = np.sum(normal_scores < threshold)
                        if count > 0:
                            self.log(f"      • 점수 < {threshold}: {count:,}개 "
                                    f"({count/len(normal_scores)*100:.2f}%)")
                
                # 이상으로 분류되었지만 점수가 높은 데이터
                if len(anomaly_scores) > 0:
                    anomaly_but_high_score = np.sum(anomaly_scores > 0)
                    if anomaly_but_high_score > 0:
                        self.log(f"    - 이상 분류지만 점수 > 0: {anomaly_but_high_score:,}개 "
                                f"({anomaly_but_high_score/len(anomaly_scores)*100:.1f}%)")
                
                # 경계 근처 데이터 분석
                boundary_range = 10  # 결정 경계 ±10
                near_boundary = np.sum(np.abs(scores - decision_boundary) < boundary_range)
                self.log(f"    - 결정 경계({decision_boundary:.2f}) ±{boundary_range} 범위: "
                        f"{near_boundary:,}개 ({near_boundary/len(scores)*100:.1f}%)")
                
                # 2차 로직 경계값 추천
                self.log("\n  💡 2차 로직 경계값 추천:")
                
                # 방법 1: 정상 데이터의 하위 퍼센타일
                normal_lower_bound = np.percentile(normal_scores, 1)  # 정상의 하위 1%
                self.log(f"    - 정상 데이터 하위 1%: {normal_lower_bound:.2f}")
                
                # 방법 2: 전체 데이터의 특정 퍼센타일
                overall_p3 = np.percentile(scores, 3)
                self.log(f"    - 전체 데이터 하위 3%: {overall_p3:.2f}")
                
                # 방법 3: 평균 - n*표준편차
                mean_minus_2std = np.mean(scores) - 2 * np.std(scores)
                mean_minus_3std = np.mean(scores) - 3 * np.std(scores)
                self.log(f"    - 평균 - 2σ: {mean_minus_2std:.2f}")
                self.log(f"    - 평균 - 3σ: {mean_minus_3std:.2f}")
                
                # 방법 4: 이상 데이터의 상위 경계
                if len(anomaly_scores) > 0:
                    anomaly_upper = np.percentile(anomaly_scores, 90)  # 이상의 상위 10%
                    self.log(f"    - 이상 데이터 상위 10%: {anomaly_upper:.2f}")
                
                # 모델 정보
                model_info = {
                    'machine_id': machine_id,
                    'sensor': sensor,
                    'train_samples': len(X_train),
                    'training_periods': self.training_periods,
                    'features': self.sensor_config[sensor]['features'],
                    'best_params': self.study.best_params,
                    'decision_boundary': float(decision_boundary),
                    'anomaly_ratio': float(anomaly_ratio),
                    'score_statistics': {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores))
                    },
                    'normal_score_statistics': {
                        'count': int(len(normal_scores)),
                        'mean': float(np.mean(normal_scores)),
                        'std': float(np.std(normal_scores)),
                        'min': float(np.min(normal_scores)),
                        'max': float(np.max(normal_scores)),
                        'percentiles': {
                            'p1': float(np.percentile(normal_scores, 1)),
                            'p5': float(np.percentile(normal_scores, 5)),
                            'p10': float(np.percentile(normal_scores, 10))
                        }
                    },
                    'anomaly_score_statistics': {
                        'count': int(len(anomaly_scores)),
                        'mean': float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else None,
                        'std': float(np.std(anomaly_scores)) if len(anomaly_scores) > 0 else None,
                        'min': float(np.min(anomaly_scores)) if len(anomaly_scores) > 0 else None,
                        'max': float(np.max(anomaly_scores)) if len(anomaly_scores) > 0 else None
                    },
                    'score_percentiles': percentile_values,
                    'score_distribution': score_distribution,
                    'secondary_thresholds': {
                        'normal_p1': float(normal_lower_bound),
                        'overall_p3': float(overall_p3),
                        'mean_minus_2std': float(mean_minus_2std),
                        'mean_minus_3std': float(mean_minus_3std),
                        'anomaly_p90': float(anomaly_upper) if len(anomaly_scores) > 0 else None
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
                
                # 시간대별 분석을 위한 데이터 준비
                self.log(f"\n{test_date} 시간대별 분석 중...")
                
                # 원본 데이터에 예측 결과 추가
                hourly_stats = self.analyze_hourly_anomalies(
                    machine_id, sensor, test_date, 
                    predictions, scores
                )
                
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
                
                # 시간대별 이상 탐지 결과
                if hourly_stats:
                    self.result_text.insert(tk.END, "\n  시간대별 이상 탐지:\n")
                    for hour, stats in hourly_stats.items():
                        if stats['anomaly_count'] > 0:
                            self.result_text.insert(
                                tk.END, 
                                f"    {hour:02d}시: {stats['anomaly_count']}개/"
                                f"{stats['total_count']}개 ({stats['anomaly_ratio']:.1f}%), "
                                f"점수: {stats['mean_score']:.2f}\n"
                            )
                    
                    # 이상이 가장 많이 발생한 시간대
                    peak_hour = max(hourly_stats.items(), 
                                  key=lambda x: x[1]['anomaly_count'])[0]
                    self.result_text.insert(tk.END, 
                                          f"  - 이상 최다 발생 시간: {peak_hour}시\n")
                
            self.result_text.insert(tk.END, f"\n{'='*60}\n")
            self.result_text.insert(tk.END, "테스트 완료\n")
            
        except Exception as e:
            messagebox.showerror("오류", f"테스트 중 오류 발생: {e}")
    
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
                if hourly_predictions[hour]:
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