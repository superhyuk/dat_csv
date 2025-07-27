#!/usr/bin/env python3
# large_csv_plotter.py - 대용량 CSV 파일 시각화 도구 (30분 단위 시간축)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QComboBox, QGroupBox, QRadioButton,
                             QButtonGroup, QSlider, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import gc
from datetime import datetime, timedelta

class DataLoader(QThread):
    """백그라운드에서 데이터를 로드하는 스레드"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame, dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_path, sample_rate=None):
        super().__init__()
        self.file_path = file_path
        self.sample_rate = sample_rate
        
    def run(self):
        try:
            # 1. 빠른 미리보기를 위해 첫 몇 줄만 읽어서 컬럼 확인
            self.progress.emit("파일 구조 확인 중...")
            preview = pd.read_csv(self.file_path, nrows=5)
            columns = preview.columns.tolist()
            
            # 센서 타입 자동 감지
            if 'x' in columns and 'y' in columns and 'z' in columns:
                sensor_type = 'ACC'
            elif 'mic_value' in columns:
                sensor_type = 'MIC'
            else:
                sensor_type = 'UNKNOWN'
            
            # 2. 청크 단위로 읽기 (메모리 효율성)
            self.progress.emit("데이터 로드 중...")
            chunk_size = 1_000_000  # 100만 행씩 읽기
            
            # 다운샘플링 비율 자동 계산
            total_rows = sum(1 for _ in open(self.file_path)) - 1  # 헤더 제외
            
            if total_rows > 10_000_000:  # 1000만개 이상
                downsample_rate = 100
                self.progress.emit(f"대용량 파일 감지: {total_rows:,}개 행 (1/100 다운샘플링)")
            elif total_rows > 1_000_000:  # 100만개 이상
                downsample_rate = 10
                self.progress.emit(f"중간 크기 파일: {total_rows:,}개 행 (1/10 다운샘플링)")
            else:
                downsample_rate = 1
                self.progress.emit(f"소규모 파일: {total_rows:,}개 행 (다운샘플링 없음)")
            
            # 3. 효율적인 데이터 로드
            if downsample_rate > 1:
                # 다운샘플링하여 읽기
                skiprows = lambda x: x % downsample_rate != 0
                df = pd.read_csv(self.file_path, skiprows=skiprows)
            else:
                # 전체 읽기
                df = pd.read_csv(self.file_path)
            
            # 4. timestamp 처리
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    self.progress.emit("timestamp 변환 실패, 인덱스 사용")
            
            # 메타데이터
            metadata = {
                'sensor_type': sensor_type,
                'total_rows': total_rows,
                'loaded_rows': len(df),
                'downsample_rate': downsample_rate,
                'columns': columns,
                'sample_rate': self.sample_rate
            }
            
            self.finished.emit(df, metadata)
            
        except Exception as e:
            self.error.emit(f"데이터 로드 실패: {str(e)}")


class PlotCanvas(FigureCanvas):
    """matplotlib 그래프를 표시하는 캔버스"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8))
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_data(self, df, metadata, plot_type='time', time_range=None):
        """데이터 플롯"""
        self.fig.clear()
        
        sensor_type = metadata['sensor_type']
        
        if sensor_type == 'ACC':
            self._plot_acc_data(df, plot_type, time_range)
        elif sensor_type == 'MIC':
            self._plot_mic_data(df, plot_type, time_range)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, '지원하지 않는 데이터 형식', 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.fig.tight_layout()
        self.draw()
    
    def _setup_time_axis(self, ax, has_timestamp=False):
        """시간 축 포맷 설정 - 30분 단위"""
        if has_timestamp:
            # 30분 간격으로 주 눈금 설정
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            # 15분 간격으로 보조 눈금 설정
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
            # 시간 포맷 설정 (시:분)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # x축 라벨 회전
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 그리드 설정
            ax.grid(True, which='major', alpha=0.5)
            ax.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    def _plot_acc_data(self, df, plot_type, time_range=None):
        """가속도계 데이터 플롯"""
        # 시간 범위 필터링
        if time_range and 'timestamp' in df.columns:
            mask = (df.index >= time_range[0]) & (df.index <= time_range[1])
            df_plot = df.loc[mask]
        else:
            df_plot = df
        
        if plot_type == 'time':
            # 시계열 플롯
            ax1 = self.fig.add_subplot(311)
            ax2 = self.fig.add_subplot(312, sharex=ax1)
            ax3 = self.fig.add_subplot(313, sharex=ax1)
            
            has_timestamp = 'timestamp' in df_plot.columns
            
            if has_timestamp:
                x_data = df_plot['timestamp']
                ax3.set_xlabel('Time')
            else:
                x_data = df_plot.index
                ax3.set_xlabel('Sample Index')
            
            ax1.plot(x_data, df_plot['x'], 'b-', linewidth=0.5)
            ax1.set_ylabel('X-axis (g)')
            
            ax2.plot(x_data, df_plot['y'], 'g-', linewidth=0.5)
            ax2.set_ylabel('Y-axis (g)')
            
            ax3.plot(x_data, df_plot['z'], 'r-', linewidth=0.5)
            ax3.set_ylabel('Z-axis (g)')
            
            ax1.set_title('Accelerometer Data')
            
            # 모든 축에 시간 포맷 적용
            for ax in [ax1, ax2, ax3]:
                self._setup_time_axis(ax, has_timestamp)
            
            # x축 라벨은 마지막 subplot에만 표시
            plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)
            plt.setp(ax2.xaxis.get_majorticklabels(), visible=False)
            
        elif plot_type == 'magnitude':
            # 진폭 플롯
            ax = self.fig.add_subplot(111)
            magnitude = np.sqrt(df_plot['x']**2 + df_plot['y']**2 + df_plot['z']**2)
            
            has_timestamp = 'timestamp' in df_plot.columns
            
            if has_timestamp:
                ax.plot(df_plot['timestamp'], magnitude, 'k-', linewidth=0.5)
                ax.set_xlabel('Time')
            else:
                ax.plot(df_plot.index, magnitude, 'k-', linewidth=0.5)
                ax.set_xlabel('Sample Index')
            
            ax.set_ylabel('Magnitude (g)')
            ax.set_title('Accelerometer Magnitude')
            
            # 시간 축 포맷 적용
            self._setup_time_axis(ax, has_timestamp)
            
        elif plot_type == 'spectrum':
            # 주파수 스펙트럼
            from scipy import signal
            
            ax = self.fig.add_subplot(111)
            
            # 각 축별 스펙트럼
            for i, (axis, color) in enumerate([('x', 'b'), ('y', 'g'), ('z', 'r')]):
                # 스펙트럼 계산
                f, Pxx = signal.welch(df_plot[axis].values, 
                                     fs=1666,  # 샘플링 레이트
                                     nperseg=min(4096, len(df_plot)//4))
                ax.semilogy(f, Pxx, color=color, label=f'{axis.upper()}-axis', alpha=0.7)
            
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title('Frequency Spectrum')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 833)  # 나이퀴스트 주파수까지
    
    def _plot_mic_data(self, df, plot_type, time_range=None):
        """마이크 데이터 플롯"""
        # 시간 범위 필터링
        if time_range and 'timestamp' in df.columns:
            mask = (df.index >= time_range[0]) & (df.index <= time_range[1])
            df_plot = df.loc[mask]
        else:
            df_plot = df
        
        if plot_type == 'time':
            # 시계열 플롯
            ax = self.fig.add_subplot(111)
            
            has_timestamp = 'timestamp' in df_plot.columns
            
            if has_timestamp:
                ax.plot(df_plot['timestamp'], df_plot['mic_value'], 'b-', linewidth=0.5)
                ax.set_xlabel('Time')
            else:
                ax.plot(df_plot.index, df_plot['mic_value'], 'b-', linewidth=0.5)
                ax.set_xlabel('Sample Index')
            
            ax.set_ylabel('Amplitude')
            ax.set_title('Microphone Data')
            
            # 시간 축 포맷 적용
            self._setup_time_axis(ax, has_timestamp)
            
        elif plot_type == 'spectrum':
            # 주파수 스펙트럼
            from scipy import signal
            
            ax = self.fig.add_subplot(111)
            
            # 스펙트럼 계산
            f, Pxx = signal.welch(df_plot['mic_value'].values, 
                                 fs=8000,  # 샘플링 레이트
                                 nperseg=min(8192, len(df_plot)//4))
            ax.semilogy(f, Pxx, 'b-')
            
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title('Microphone Frequency Spectrum')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 4000)  # 나이퀴스트 주파수까지
            
        elif plot_type == 'spectrogram':
            # 스펙트로그램
            from scipy import signal
            
            ax = self.fig.add_subplot(111)
            
            # 스펙트로그램 계산
            f, t, Sxx = signal.spectrogram(df_plot['mic_value'].values, 
                                          fs=8000,
                                          nperseg=min(512, len(df_plot)//8))
            
            # 로그 스케일로 변환
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            # 시간 축을 timestamp로 변환 (있는 경우)
            if 'timestamp' in df_plot.columns:
                # 시작 시간
                start_time = df_plot['timestamp'].iloc[0]
                # 시간 배열을 datetime으로 변환
                time_stamps = [start_time + timedelta(seconds=float(t_val)) for t_val in t]
                
                im = ax.pcolormesh(mdates.date2num(time_stamps), f, Sxx_db, 
                                  shading='gouraud', cmap='viridis')
                
                # 시간 축 포맷 적용
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                ax.set_xlabel('Time')
            else:
                im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
                ax.set_xlabel('Time (s)')
            
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('Microphone Spectrogram')
            ax.set_ylim(0, 4000)
            
            # 컬러바 추가
            cbar = self.fig.colorbar(im, ax=ax)
            cbar.set_label('Power (dB)')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("대용량 CSV 파일 플로터 - 30분 단위 시간축")
        self.setGeometry(100, 100, 1400, 900)
        
        # 데이터 저장
        self.df = None
        self.metadata = None
        
        # UI 초기화
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 상단 컨트롤
        control_layout = QHBoxLayout()
        
        # 파일 선택
        self.file_label = QLabel("파일을 선택하세요")
        control_layout.addWidget(self.file_label)
        
        self.load_button = QPushButton("CSV 파일 열기")
        self.load_button.clicked.connect(self.load_file)
        control_layout.addWidget(self.load_button)
        
        control_layout.addStretch()
        
        # 플롯 타입 선택
        plot_group = QGroupBox("플롯 유형")
        plot_layout = QHBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['시계열', '진폭', '주파수 스펙트럼', '스펙트로그램'])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)
        plot_layout.addWidget(self.plot_type_combo)
        
        plot_group.setLayout(plot_layout)
        control_layout.addWidget(plot_group)
        
        # 시간 범위 선택
        range_group = QGroupBox("시간 범위")
        range_layout = QHBoxLayout()
        
        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setMinimum(0)
        self.range_slider.setMaximum(100)
        self.range_slider.setValue(100)
        self.range_slider.valueChanged.connect(self.update_time_range)
        range_layout.addWidget(self.range_slider)
        
        self.range_label = QLabel("100%")
        range_layout.addWidget(self.range_label)
        
        range_group.setLayout(range_layout)
        control_layout.addWidget(range_group)
        
        layout.addLayout(control_layout)
        
        # 정보 표시
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)
        
        # 그래프 캔버스
        self.canvas = PlotCanvas(self)
        layout.addWidget(self.canvas)
        
        # 상태 표시
        self.status_label = QLabel("준비")
        self.statusBar().addWidget(self.status_label)
    
    def load_file(self):
        """CSV 파일 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "CSV 파일 선택", "", "CSV 파일 (*.csv);;모든 파일 (*.*)"
        )
        
        if file_path:
            self.file_label.setText(f"로딩 중: {file_path}")
            self.load_button.setEnabled(False)
            
            # 백그라운드에서 로드
            self.loader = DataLoader(file_path)
            self.loader.progress.connect(self.update_status)
            self.loader.finished.connect(self.on_data_loaded)
            self.loader.error.connect(self.on_load_error)
            self.loader.start()
    
    def on_data_loaded(self, df, metadata):
        """데이터 로드 완료"""
        self.df = df
        self.metadata = metadata
        
        # UI 업데이트
        self.file_label.setText(f"로드 완료: {metadata['loaded_rows']:,}개 행")
        self.load_button.setEnabled(True)
        
        # 정보 표시
        info_text = f"센서 타입: {metadata['sensor_type']} | "
        info_text += f"전체 행: {metadata['total_rows']:,} | "
        info_text += f"로드된 행: {metadata['loaded_rows']:,} | "
        if metadata['downsample_rate'] > 1:
            info_text += f"다운샘플링: 1/{metadata['downsample_rate']}"
        
        # 시간 범위 정보 추가
        if 'timestamp' in df.columns:
            start_time = df['timestamp'].min()
            end_time = df['timestamp'].max()
            duration = end_time - start_time
            info_text += f" | 시간 범위: {start_time.strftime('%H:%M')} ~ {end_time.strftime('%H:%M')} ({duration})"
        
        self.info_label.setText(info_text)
        
        # 플롯 타입 업데이트
        self.plot_type_combo.clear()
        if metadata['sensor_type'] == 'ACC':
            self.plot_type_combo.addItems(['시계열', '진폭', '주파수 스펙트럼'])
        elif metadata['sensor_type'] == 'MIC':
            self.plot_type_combo.addItems(['시계열', '주파수 스펙트럼', '스펙트로그램'])
        
        # 초기 플롯
        self.update_plot()
    
    def on_load_error(self, error_msg):
        """로드 에러 처리"""
        self.status_label.setText(f"에러: {error_msg}")
        self.load_button.setEnabled(True)
        self.file_label.setText("파일을 선택하세요")
    
    def update_status(self, message):
        """상태 업데이트"""
        self.status_label.setText(message)
    
    def update_plot(self):
        """플롯 업데이트"""
        if self.df is None:
            return
        
        plot_type_map = {
            '시계열': 'time',
            '진폭': 'magnitude',
            '주파수 스펙트럼': 'spectrum',
            '스펙트로그램': 'spectrogram'
        }
        
        plot_type = plot_type_map.get(self.plot_type_combo.currentText(), 'time')
        
        # 시간 범위 계산
        range_percent = self.range_slider.value() / 100
        if range_percent < 1.0:
            total_samples = len(self.df)
            end_idx = int(total_samples * range_percent)
            time_range = (0, end_idx)
        else:
            time_range = None
        
        self.canvas.plot_data(self.df, self.metadata, plot_type, time_range)
        self.update_status("플롯 완료")
    
    def update_time_range(self, value):
        """시간 범위 슬라이더 업데이트"""
        self.range_label.setText(f"{value}%")
        if self.df is not None:
            self.update_plot()


class QuickPlotter:
    """간단한 정적 플롯 함수들"""
    
    @staticmethod
    def plot_csv_quick(file_path, max_points=100000):
        """CSV 파일을 빠르게 플롯하는 간단한 함수"""
        # 데이터 로드
        df = pd.read_csv(file_path)
        
        # timestamp 변환
        has_timestamp = False
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                has_timestamp = True
            except:
                pass
        
        # 다운샘플링
        if len(df) > max_points:
            step = len(df) // max_points
            df = df.iloc[::step]
        
        # 센서 타입 감지
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            # ACC 데이터
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            
            x_data = df['timestamp'] if has_timestamp else df.index
            
            axes[0].plot(x_data, df['x'], 'b-', linewidth=0.5)
            axes[0].set_ylabel('X (g)')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(x_data, df['y'], 'g-', linewidth=0.5)
            axes[1].set_ylabel('Y (g)')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(x_data, df['z'], 'r-', linewidth=0.5)
            axes[2].set_ylabel('Z (g)')
            axes[2].grid(True, alpha=0.3)
            
            if has_timestamp:
                axes[2].set_xlabel('Time')
                # 30분 간격 설정
                for ax in axes:
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                
                # x축 라벨 회전
                plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                axes[2].set_xlabel('Sample')
            
            plt.suptitle('Accelerometer Data')
            
        elif 'mic_value' in df.columns:
            # MIC 데이터
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x_data = df['timestamp'] if has_timestamp else df.index
            ax.plot(x_data, df['mic_value'], 'b-', linewidth=0.5)
            
            if has_timestamp:
                ax.set_xlabel('Time')
                # 30분 간격 설정
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.set_xlabel('Sample')
            
            ax.set_ylabel('Amplitude')
            ax.set_title('Microphone Data')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    @staticmethod
    def plot_time_range(file_path, start_time, end_time, time_column='timestamp'):
        """특정 시간 범위만 플롯"""
        # 시간 컬럼만 먼저 읽어서 범위 확인
        time_df = pd.read_csv(file_path, usecols=[time_column])
        time_df[time_column] = pd.to_datetime(time_df[time_column])
        
        # 해당 범위의 인덱스 찾기
        mask = (time_df[time_column] >= start_time) & (time_df[time_column] <= end_time)
        indices = time_df.index[mask].tolist()
        
        if not indices:
            print("해당 시간 범위에 데이터가 없습니다.")
            return None
        
        # 해당 범위만 읽기
        df = pd.read_csv(file_path, skiprows=lambda x: x != 0 and x not in indices)
        df[time_column] = pd.to_datetime(df[time_column])
        
        # 플롯
        if 'x' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df[time_column], df['x'], 'b-', label='X', linewidth=0.5)
            ax.plot(df[time_column], df['y'], 'g-', label='Y', linewidth=0.5)
            ax.plot(df[time_column], df['z'], 'r-', label='Z', linewidth=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Acceleration (g)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 30분 간격 설정
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        return df


# 사용 예제
if __name__ == "__main__":
    # GUI 버전
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
    # 또는 간단한 버전
    # QuickPlotter.plot_csv_quick("your_file.csv")
    # QuickPlotter.plot_time_range("your_file.csv", "2025-07-24 09:00:00", "2025-07-24 10:00:00")