#!/usr/bin/env python3
# dat_to_csv_gui.py - DAT → CSV 변환 GUI 프로그램

import os
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QRadioButton, QButtonGroup, QFileDialog, QSpinBox, 
                             QTextEdit, QGroupBox, QFormLayout, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class ConversionWorker(QThread):
    """변환 작업을 백그라운드에서 실행하는 클래스"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, sensor_type, dat_paths, sampling_rate, window_sec, is_directory=True):
        super().__init__()
        self.sensor_type = sensor_type
        self.dat_paths = dat_paths  # 폴더 경로 또는 파일 경로 리스트
        self.sampling_rate = sampling_rate
        self.window_sec = window_sec
        self.is_directory = is_directory  # 폴더 모드 여부
        
    def load_dat_file_acc(self, dat_path, sampling_rate, window_sec):
        """ACC용 DAT을 읽어 float64 2D np.ndarray (N×3) 반환"""
        samples_needed = sampling_rate * window_sec
        data = np.empty((samples_needed, 3), dtype=np.int16)
        with open(dat_path, 'rb') as f:
            sets = (samples_needed + 999) // 1000
            total = 0
            for i in range(sets):
                chunk = np.fromfile(f, dtype=np.int16, count=3000)
                if len(chunk) < 3000:
                    raise ValueError(f"불완전 DAT: 청크{i}에서 {len(chunk)}개 읽음")
                chunk = chunk.reshape(-1, 3)
                csize = min(1000, samples_needed - total)
                data[total:total+csize, :] = chunk[:csize, :]
                total += csize
                f.seek(8, 1)
        return data.astype(np.float64) * 0.000488  # 스케일링
    
    def load_dat_file_mic(self, dat_path, sampling_rate, window_sec):
        """MIC용 DAT을 읽어 float64 1D np.ndarray 반환"""
        samples_needed = sampling_rate * window_sec
        buf = np.empty(samples_needed, dtype=np.int16)
        with open(dat_path, 'rb') as f:
            sets = samples_needed // 1000
            for i in range(sets):
                chunk = np.fromfile(f, dtype=np.int16, count=1000)
                if len(chunk) < 1000:
                    raise ValueError(f"불완전 DAT: 청크{i}에서 {len(chunk)}개 읽음")
                buf[i*1000:(i+1)*1000] = chunk
                f.seek(8, 1)  # 8바이트 스킵
        return buf.astype(np.int16)
    
    def run(self):
        """변환 작업 실행"""
        try:
            success_count = 0
            fail_count = 0
            
            if self.is_directory:
                # 폴더 모드 - 재귀적으로 처리
                dat_dir = self.dat_paths
                self.progress_signal.emit(f"▶️ {self.sensor_type} 변환 시작 (재귀 탐색): {dat_dir}")
                
                for root, _, files in os.walk(dat_dir):
                    dats = [f for f in files if f.lower().endswith('.dat')]
                    if not dats:
                        continue
                    
                    csv_dir = os.path.join(root, 'csv')
                    os.makedirs(csv_dir, exist_ok=True)
                    
                    for fname in sorted(dats):
                        dat_path = os.path.join(root, fname)
                        try:
                            if self.sensor_type == "ACC":
                                data = self.load_dat_file_acc(dat_path, self.sampling_rate, self.window_sec)
                                n = data.shape[0]
                                times = np.arange(n) / self.sampling_rate
                                out = np.column_stack((times, data))
                                header = 'time_sec,x,y,z'
                                np.savetxt(os.path.join(csv_dir, os.path.splitext(fname)[0] + '.csv'), out, delimiter=',', header=header, comments='', fmt=['%.6f', '%.6f', '%.6f', '%.6f'])
                            else:  # MIC
                                data = self.load_dat_file_mic(dat_path, self.sampling_rate, self.window_sec)
                                n = data.shape[0]
                                times = np.arange(n) / self.sampling_rate
                                out = np.column_stack((times, data))
                                header = 'time_sec,mic_value'
                                # time은 소수점 6자리까지, mic 값은 정수형으로 저장
                                np.savetxt(os.path.join(csv_dir, os.path.splitext(fname)[0] + '.csv'), out, delimiter=',', header=header, comments='', fmt=['%.6f', '%d'])
                            success_count += 1
                            self.progress_signal.emit(f"   ✅ {os.path.relpath(dat_path, dat_dir)} → {os.path.relpath(os.path.join(csv_dir, os.path.splitext(fname)[0] + '.csv'), dat_dir)}")
                        except Exception as e:
                            fail_count += 1
                            self.progress_signal.emit(f"   ❌ 실패: {os.path.relpath(dat_path, dat_dir)} ({e})")
            else:
                # 파일 모드 - 선택된 파일들만 처리
                self.progress_signal.emit(f"▶️ {self.sensor_type} 변환 시작 (선택된 파일): {len(self.dat_paths)}개")
                
                for dat_path in self.dat_paths:
                    try:
                        # 출력 폴더는 입력 파일과 같은 위치의 'csv' 폴더
                        file_dir = os.path.dirname(dat_path)
                        csv_dir = os.path.join(file_dir, 'csv')
                        os.makedirs(csv_dir, exist_ok=True)
                        
                        fname = os.path.basename(dat_path)
                        
                        if self.sensor_type == "ACC":
                            data = self.load_dat_file_acc(dat_path, self.sampling_rate, self.window_sec)
                            n = data.shape[0]
                            times = np.arange(n) / self.sampling_rate
                            out = np.column_stack((times, data))
                            header = 'time_sec,x,y,z'
                            np.savetxt(os.path.join(csv_dir, os.path.splitext(fname)[0] + '.csv'), out, delimiter=',', header=header, comments='', fmt=['%.6f', '%.6f', '%.6f', '%.6f'])
                        else:  # MIC
                            data = self.load_dat_file_mic(dat_path, self.sampling_rate, self.window_sec)
                            n = data.shape[0]
                            times = np.arange(n) / self.sampling_rate
                            out = np.column_stack((times, data))
                            header = 'time_sec,mic_value'
                            # time은 소수점 6자리까지, mic 값은 정수형으로 저장
                            np.savetxt(os.path.join(csv_dir, os.path.splitext(fname)[0] + '.csv'), out, delimiter=',', header=header, comments='', fmt=['%.6f', '%d'])
                        success_count += 1
                        self.progress_signal.emit(f"   ✅ {fname} → {os.path.join('csv', os.path.splitext(fname)[0] + '.csv')}")
                    except Exception as e:
                        fail_count += 1
                        self.progress_signal.emit(f"   ❌ 실패: {fname} ({e})")
            
            mode_str = "폴더" if self.is_directory else "파일"
            self.finished_signal.emit(True, f"✅ {self.sensor_type} 전체 변환 완료 ({mode_str} 모드): 성공 {success_count}개, 실패 {fail_count}개")
        except Exception as e:
            self.finished_signal.emit(False, f"❌ 변환 중 오류 발생: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAT → CSV 변환기")
        self.setMinimumSize(700, 550)
        
        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # ===== 센서 유형 선택 =====
        sensor_group = QGroupBox("센서 유형")
        sensor_layout = QHBoxLayout()
        
        self.acc_radio = QRadioButton("가속도계 (ACC)")
        self.mic_radio = QRadioButton("마이크 (MIC)")
        self.acc_radio.setChecked(True)  # 기본값
        
        self.sensor_group = QButtonGroup()
        self.sensor_group.addButton(self.acc_radio)
        self.sensor_group.addButton(self.mic_radio)
        
        sensor_layout.addWidget(self.acc_radio)
        sensor_layout.addWidget(self.mic_radio)
        sensor_group.setLayout(sensor_layout)
        main_layout.addWidget(sensor_group)
        
        # ===== 모드 선택 =====
        mode_group = QGroupBox("변환 모드")
        mode_layout = QHBoxLayout()
        
        self.folder_radio = QRadioButton("폴더 모드 (하위 폴더 모두 처리)")
        self.file_radio = QRadioButton("파일 모드 (선택한 파일만 처리)")
        self.folder_radio.setChecked(True)  # 기본값
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.folder_radio)
        self.mode_group.addButton(self.file_radio)
        
        mode_layout.addWidget(self.folder_radio)
        mode_layout.addWidget(self.file_radio)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # ===== 설정 =====
        settings_group = QGroupBox("설정")
        settings_layout = QFormLayout()
        
        # 입력 선택 (폴더 모드)
        self.folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_button = QPushButton("폴더 찾기...")
        self.folder_layout.addWidget(self.folder_input)
        self.folder_layout.addWidget(self.folder_button)
        self.folder_row = settings_layout.addRow("DAT 파일 폴더:", self.folder_layout)
        
        # 입력 선택 (파일 모드)
        self.file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        self.file_button = QPushButton("파일 선택...")
        self.file_layout.addWidget(self.file_input)
        self.file_layout.addWidget(self.file_button)
        self.file_row = settings_layout.addRow("DAT 파일:", self.file_layout)
        
        # 샘플링 레이트
        self.rate_input = QSpinBox()
        self.rate_input.setRange(100, 50000)
        self.rate_input.setValue(1666)  # ACC 기본값
        settings_layout.addRow("샘플링 레이트 (Hz):", self.rate_input)
        
        # 윈도우 크기
        self.window_input = QSpinBox()
        self.window_input.setRange(1, 60)
        self.window_input.setValue(5)
        settings_layout.addRow("윈도우 크기 (초):", self.window_input)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # ===== 실행 버튼 =====
        self.convert_button = QPushButton("변환 시작")
        self.convert_button.setMinimumHeight(40)
        main_layout.addWidget(self.convert_button)
        
        # ===== 진행 상황 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # ===== 로그 영역 =====
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # 파일 모드일 때 선택된 파일 목록
        self.selected_files = []
        
        # 이벤트 연결
        self.folder_button.clicked.connect(self.browse_directory)
        self.file_button.clicked.connect(self.browse_files)
        self.convert_button.clicked.connect(self.start_conversion)
        self.acc_radio.toggled.connect(self.update_sampling_rate)
        self.mic_radio.toggled.connect(self.update_sampling_rate)
        self.folder_radio.toggled.connect(self.update_mode)
        self.file_radio.toggled.connect(self.update_mode)
        
        # 초기 상태 설정
        self.update_mode()
        
        self.worker = None

    def browse_directory(self):
        """폴더 찾아보기 대화상자 표시"""
        dir_path = QFileDialog.getExistingDirectory(self, "DAT 파일 폴더 선택")
        if dir_path:
            self.folder_input.setText(dir_path)
    
    def browse_files(self):
        """파일 찾아보기 대화상자 표시"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "DAT 파일 선택", "", "DAT 파일 (*.dat);;모든 파일 (*.*)"
        )
        if file_paths:
            self.selected_files = file_paths
            if len(file_paths) == 1:
                self.file_input.setText(os.path.basename(file_paths[0]))
            else:
                self.file_input.setText(f"{len(file_paths)}개 파일 선택됨")
    
    def update_mode(self):
        """선택된 모드에 따라 UI 업데이트"""
        is_folder_mode = self.folder_radio.isChecked()
        
        # 폴더 관련 위젯 활성화/비활성화
        self.folder_input.setEnabled(is_folder_mode)
        self.folder_button.setEnabled(is_folder_mode)
        
        # 파일 관련 위젯 활성화/비활성화
        self.file_input.setEnabled(not is_folder_mode)
        self.file_button.setEnabled(not is_folder_mode)
    
    def update_sampling_rate(self):
        """센서 유형에 따라 샘플링 레이트 기본값 업데이트"""
        if self.acc_radio.isChecked():
            self.rate_input.setValue(1666)
        else:
            self.rate_input.setValue(8000)
    
    def log_message(self, message):
        """로그 메시지 추가"""
        self.log_text.append(message)
        # 자동 스크롤
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def conversion_finished(self, success, message):
        """변환 작업 완료 처리"""
        self.log_message(message)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100 if success else 0)
        self.convert_button.setEnabled(True)
        self.convert_button.setText("변환 시작")
    
    def start_conversion(self):
        """변환 작업 시작"""
        # 센서 유형 확인
        sensor_type = "ACC" if self.acc_radio.isChecked() else "MIC"
        sampling_rate = self.rate_input.value()
        window_sec = self.window_input.value()
        is_folder_mode = self.folder_radio.isChecked()
        
        if is_folder_mode:
            # 폴더 모드 - 입력 검증
            dat_dir = self.folder_input.text().strip()
            if not dat_dir or not os.path.isdir(dat_dir):
                self.log_message("❌ 오류: 유효한 DAT 파일 폴더를 선택하세요.")
                return
            dat_paths = dat_dir
        else:
            # 파일 모드 - 입력 검증
            if not self.selected_files:
                self.log_message("❌ 오류: 하나 이상의 DAT 파일을 선택하세요.")
                return
            dat_paths = self.selected_files
        
        # UI 업데이트
        self.convert_button.setEnabled(False)
        self.convert_button.setText("변환 중...")
        self.progress_bar.setMaximum(0)  # 불확정 진행 상태
        self.log_text.clear()
        
        # 작업 스레드 시작
        self.worker = ConversionWorker(
            sensor_type, dat_paths, sampling_rate, window_sec, is_directory=is_folder_mode
        )
        self.worker.progress_signal.connect(self.log_message)
        self.worker.finished_signal.connect(self.conversion_finished)
        self.worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())