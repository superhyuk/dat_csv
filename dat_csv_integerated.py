#!/usr/bin/env python3
# dat_to_integrated_csv_gui.py - DAT → 통합 CSV 변환 GUI 프로그램

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QRadioButton, QButtonGroup, QFileDialog, QSpinBox, 
                             QTextEdit, QGroupBox, QFormLayout, QProgressBar,
                             QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class ConversionWorker(QThread):
    """변환 작업을 백그라운드에서 실행하는 클래스"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_update = pyqtSignal(int, int)  # 현재, 전체
    
    def __init__(self, sensor_type, dat_dir, sampling_rate, integration_mode):
        super().__init__()
        self.sensor_type = sensor_type
        self.dat_dir = dat_dir
        self.sampling_rate = sampling_rate
        self.integration_mode = integration_mode  # 'none', 'daily', 'all'
        
    def load_dat_file_acc(self, dat_path, sampling_rate):
        """ACC용 DAT 파일 전체를 읽어 float64 2D np.ndarray (N×3) 반환"""
        # 파일 크기로 샘플 수 계산
        file_size = os.path.getsize(dat_path)
        # 각 세트는 3000개 int16 (6000바이트) + 8바이트 스킵 = 6008바이트
        sets = file_size // 6008
        samples = sets * 1000
        
        data = np.empty((samples, 3), dtype=np.int16)
        with open(dat_path, 'rb') as f:
            for i in range(sets):
                chunk = np.fromfile(f, dtype=np.int16, count=3000)
                if len(chunk) < 3000:
                    # 마지막 청크가 불완전한 경우
                    actual_samples = len(chunk) // 3
                    if actual_samples > 0:
                        chunk = chunk[:actual_samples * 3].reshape(-1, 3)
                        data[i*1000:i*1000+actual_samples, :] = chunk
                        samples = i*1000 + actual_samples
                    else:
                        samples = i*1000
                    break
                chunk = chunk.reshape(-1, 3)
                data[i*1000:(i+1)*1000, :] = chunk
                f.seek(8, 1)  # 8바이트 스킵
        
        # 실제 읽은 샘플 수만큼 자르기
        data = data[:samples, :]
        return data.astype(np.float64) * 0.000488  # 스케일링
    
    def load_dat_file_mic(self, dat_path, sampling_rate):
        """MIC용 DAT 파일 전체를 읽어 int16 1D np.ndarray 반환"""
        # 파일 크기로 샘플 수 계산
        file_size = os.path.getsize(dat_path)
        # 각 세트는 1000개 int16 (2000바이트) + 8바이트 스킵 = 2008바이트
        sets = file_size // 2008
        samples = sets * 1000
        
        buf = np.empty(samples, dtype=np.int16)
        with open(dat_path, 'rb') as f:
            for i in range(sets):
                chunk = np.fromfile(f, dtype=np.int16, count=1000)
                if len(chunk) < 1000:
                    # 마지막 청크가 불완전한 경우
                    if len(chunk) > 0:
                        buf[i*1000:i*1000+len(chunk)] = chunk
                        samples = i*1000 + len(chunk)
                    else:
                        samples = i*1000
                    break
                buf[i*1000:(i+1)*1000] = chunk
                f.seek(8, 1)  # 8바이트 스킵
        
        # 실제 읽은 샘플 수만큼 자르기
        return buf[:samples]
    
    def extract_datetime_from_filename(self, filename):
        """파일명에서 날짜/시간 추출 (YYYYMMDD_HH_MM_SS_...)"""
        try:
            parts = filename.split('_')
            if len(parts) >= 4:
                date_str = parts[0]
                hour = parts[1]
                minute = parts[2]
                second = parts[3]
                
                # YYYYMMDD 형식 확인
                if len(date_str) == 8 and date_str.isdigit():
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    
                    # 시간 검증
                    if (hour.isdigit() and minute.isdigit() and second.isdigit() and
                        0 <= int(hour) <= 23 and 0 <= int(minute) <= 59 and 0 <= int(second) <= 59):
                        
                        dt = datetime(year, month, day, int(hour), int(minute), int(second))
                        return dt
        except:
            pass
        return None
    
    def run(self):
        """변환 작업 실행"""
        try:
            self.progress_signal.emit(f"▶️ {self.sensor_type} 변환 시작 (통합 모드: {self.integration_mode})")
            
            # 모든 DAT 파일 찾기
            all_dat_files = []
            for root, _, files in os.walk(self.dat_dir):
                dats = [os.path.join(root, f) for f in files if f.lower().endswith('.dat')]
                all_dat_files.extend(sorted(dats))
            
            if not all_dat_files:
                self.finished_signal.emit(False, "❌ DAT 파일을 찾을 수 없습니다.")
                return
            
            self.progress_signal.emit(f"📁 총 {len(all_dat_files)}개 DAT 파일 발견")
            
            if self.integration_mode == 'none':
                # 기존 방식 - 개별 CSV 생성
                self._convert_individual_files(all_dat_files)
            elif self.integration_mode == 'daily':
                # 날짜별 통합
                self._convert_daily_integrated(all_dat_files)
            else:  # 'all'
                # 전체 통합
                self._convert_all_integrated(all_dat_files)
                
        except Exception as e:
            self.finished_signal.emit(False, f"❌ 변환 중 오류 발생: {str(e)}")
    
    def _convert_individual_files(self, dat_files):
        """개별 파일로 변환 (기존 방식)"""
        success_count = 0
        fail_count = 0
        total_files = len(dat_files)
        
        for idx, dat_path in enumerate(dat_files):
            self.progress_update.emit(idx + 1, total_files)
            
            try:
                file_dir = os.path.dirname(dat_path)
                fname = os.path.basename(dat_path)
                csv_dir = os.path.join(file_dir, 'csv')
                os.makedirs(csv_dir, exist_ok=True)
                
                if self.sensor_type == "ACC":
                    data = self.load_dat_file_acc(dat_path, self.sampling_rate)
                    n = data.shape[0]
                    times = np.arange(n) / self.sampling_rate
                    out = np.column_stack((times, data))
                    header = 'time_sec,x,y,z'
                    np.savetxt(os.path.join(csv_dir, os.path.splitext(fname)[0] + '.csv'), 
                              out, delimiter=',', header=header, comments='', 
                              fmt=['%.6f', '%.6f', '%.6f', '%.6f'])
                else:  # MIC
                    data = self.load_dat_file_mic(dat_path, self.sampling_rate)
                    n = data.shape[0]
                    times = np.arange(n) / self.sampling_rate
                    out = np.column_stack((times, data))
                    header = 'time_sec,mic_value'
                    np.savetxt(os.path.join(csv_dir, os.path.splitext(fname)[0] + '.csv'), 
                              out, delimiter=',', header=header, comments='', 
                              fmt=['%.6f', '%d'])
                
                success_count += 1
                self.progress_signal.emit(f"   ✅ {os.path.relpath(dat_path, self.dat_dir)}")
                
            except Exception as e:
                fail_count += 1
                self.progress_signal.emit(f"   ❌ 실패: {os.path.relpath(dat_path, self.dat_dir)} ({e})")
        
        self.finished_signal.emit(True, f"✅ 개별 변환 완료: 성공 {success_count}개, 실패 {fail_count}개")
    
    def _convert_daily_integrated(self, dat_files):
        """날짜별 통합 변환"""
        # 날짜별로 파일 그룹화
        daily_files = {}
        for dat_path in dat_files:
            fname = os.path.basename(dat_path)
            dt = self.extract_datetime_from_filename(fname)
            if dt:
                date_key = dt.strftime('%Y-%m-%d')
                if date_key not in daily_files:
                    daily_files[date_key] = []
                daily_files[date_key].append((dt, dat_path))
        
        self.progress_signal.emit(f"📅 {len(daily_files)}개 날짜로 그룹화됨")
        
        success_days = 0
        fail_days = 0
        total_days = len(daily_files)
        current_day = 0
        
        # CSV 출력 폴더
        csv_dir = os.path.join(self.dat_dir, 'integrated_csv')
        os.makedirs(csv_dir, exist_ok=True)
        
        for date_key, file_list in sorted(daily_files.items()):
            current_day += 1
            self.progress_update.emit(current_day, total_days)
            
            try:
                # 시간순으로 정렬
                file_list.sort(key=lambda x: x[0])
                
                self.progress_signal.emit(f"\n📆 {date_key} 처리 중 ({len(file_list)}개 파일)...")
                
                all_data = []
                
                for dt, dat_path in file_list:
                    try:
                        if self.sensor_type == "ACC":
                            data = self.load_dat_file_acc(dat_path, self.sampling_rate)
                            df = pd.DataFrame(data, columns=['x', 'y', 'z'])
                        else:  # MIC
                            data = self.load_dat_file_mic(dat_path, self.sampling_rate)
                            df = pd.DataFrame(data, columns=['mic_value'])
                        
                        # 절대 시간 계산
                        time_offset = np.arange(len(df)) / self.sampling_rate
                        df['timestamp'] = pd.to_datetime(dt) + pd.to_timedelta(time_offset, unit='s')
                        df['filename'] = os.path.basename(dat_path)
                        
                        all_data.append(df)
                        
                    except Exception as e:
                        self.progress_signal.emit(f"   ⚠️ 파일 스킵: {os.path.basename(dat_path)} ({e})")
                
                if all_data:
                    # 데이터 결합
                    combined_df = pd.concat(all_data, ignore_index=True)
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                    
                    # CSV 저장
                    output_file = os.path.join(csv_dir, f"{date_key}_{self.sensor_type}_integrated.csv")
                    combined_df.to_csv(output_file, index=False)
                    
                    success_days += 1
                    self.progress_signal.emit(f"   ✅ {date_key} 통합 완료: {len(combined_df):,}개 샘플")
                    self.progress_signal.emit(f"      → {output_file}")
                else:
                    fail_days += 1
                    self.progress_signal.emit(f"   ❌ {date_key} 실패: 유효한 데이터 없음")
                    
            except Exception as e:
                fail_days += 1
                self.progress_signal.emit(f"   ❌ {date_key} 실패: {e}")
        
        self.finished_signal.emit(True, 
            f"✅ 날짜별 통합 완료: 성공 {success_days}일, 실패 {fail_days}일\n" +
            f"   저장 위치: {csv_dir}")
    
    def _convert_all_integrated(self, dat_files):
        """전체 통합 변환"""
        self.progress_signal.emit("🔄 전체 파일을 하나로 통합 중...")
        
        all_data = []
        success_count = 0
        fail_count = 0
        total_files = len(dat_files)
        
        for idx, dat_path in enumerate(dat_files):
            self.progress_update.emit(idx + 1, total_files)
            
            fname = os.path.basename(dat_path)
            dt = self.extract_datetime_from_filename(fname)
            
            try:
                if self.sensor_type == "ACC":
                    data = self.load_dat_file_acc(dat_path, self.sampling_rate)
                    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
                else:  # MIC
                    data = self.load_dat_file_mic(dat_path, self.sampling_rate)
                    df = pd.DataFrame(data, columns=['mic_value'])
                
                # 시간 정보 추가
                if dt:
                    time_offset = np.arange(len(df)) / self.sampling_rate
                    df['timestamp'] = pd.to_datetime(dt) + pd.to_timedelta(time_offset, unit='s')
                else:
                    # 날짜를 파싱할 수 없는 경우 상대 시간만 사용
                    df['timestamp'] = np.arange(len(df)) / self.sampling_rate
                
                df['filename'] = fname
                all_data.append(df)
                success_count += 1
                
                if success_count % 100 == 0:
                    self.progress_signal.emit(f"   처리 중: {success_count}/{total_files} 파일...")
                    
            except Exception as e:
                fail_count += 1
                self.progress_signal.emit(f"   ⚠️ 파일 스킵: {fname} ({e})")
        
        if all_data:
            self.progress_signal.emit("📊 데이터 결합 중...")
            
            # 모든 데이터 결합
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # timestamp가 datetime인 경우만 정렬
            if pd.api.types.is_datetime64_any_dtype(combined_df['timestamp']):
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # CSV 저장
            csv_dir = os.path.join(self.dat_dir, 'integrated_csv')
            os.makedirs(csv_dir, exist_ok=True)
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(csv_dir, f"ALL_{self.sensor_type}_integrated_{timestamp_str}.csv")
            
            self.progress_signal.emit("💾 CSV 파일 저장 중...")
            combined_df.to_csv(output_file, index=False)
            
            self.finished_signal.emit(True, 
                f"✅ 전체 통합 완료:\n" +
                f"   - 성공: {success_count}개 파일\n" +
                f"   - 실패: {fail_count}개 파일\n" +
                f"   - 총 샘플 수: {len(combined_df):,}개\n" +
                f"   - 저장 위치: {output_file}")
        else:
            self.finished_signal.emit(False, "❌ 유효한 데이터를 찾을 수 없습니다.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAT → 통합 CSV 변환기")
        self.setMinimumSize(800, 650)
        
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
        
        # ===== 통합 모드 선택 =====
        integration_group = QGroupBox("통합 모드")
        integration_layout = QVBoxLayout()
        
        self.integration_combo = QComboBox()
        self.integration_combo.addItem("개별 파일 (기존 방식)", "none")
        self.integration_combo.addItem("날짜별 통합 (하루씩 통합)", "daily")
        self.integration_combo.addItem("전체 통합 (모든 파일을 하나로)", "all")
        self.integration_combo.setCurrentIndex(1)  # 기본값: 날짜별 통합
        
        integration_layout.addWidget(self.integration_combo)
        integration_group.setLayout(integration_layout)
        main_layout.addWidget(integration_group)
        
        # ===== 설정 =====
        settings_group = QGroupBox("설정")
        settings_layout = QFormLayout()
        
        # 입력 폴더 선택
        self.folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_button = QPushButton("폴더 찾기...")
        self.folder_layout.addWidget(self.folder_input)
        self.folder_layout.addWidget(self.folder_button)
        settings_layout.addRow("DAT 파일 폴더:", self.folder_layout)
        
        # 샘플링 레이트
        self.rate_input = QSpinBox()
        self.rate_input.setRange(100, 50000)
        self.rate_input.setValue(1666)  # ACC 기본값
        settings_layout.addRow("샘플링 레이트 (Hz):", self.rate_input)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # ===== 실행 버튼 =====
        self.convert_button = QPushButton("변환 시작")
        self.convert_button.setMinimumHeight(40)
        main_layout.addWidget(self.convert_button)
        
        # ===== 진행 상황 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)
        
        # ===== 로그 영역 =====
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # 이벤트 연결
        self.folder_button.clicked.connect(self.browse_directory)
        self.convert_button.clicked.connect(self.start_conversion)
        self.acc_radio.toggled.connect(self.update_sampling_rate)
        self.mic_radio.toggled.connect(self.update_sampling_rate)
        
        self.worker = None

    def browse_directory(self):
        """폴더 찾아보기 대화상자 표시"""
        dir_path = QFileDialog.getExistingDirectory(self, "DAT 파일 폴더 선택")
        if dir_path:
            self.folder_input.setText(dir_path)
    
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
    
    def update_progress(self, current, total):
        """진행률 업데이트"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} ({current/total*100:.1f}%)")
    
    def conversion_finished(self, success, message):
        """변환 작업 완료 처리"""
        self.log_message("")
        self.log_message(message)
        if success:
            self.progress_bar.setValue(self.progress_bar.maximum())
        else:
            self.progress_bar.setValue(0)
        self.convert_button.setEnabled(True)
        self.convert_button.setText("변환 시작")
    
    def start_conversion(self):
        """변환 작업 시작"""
        # 입력 검증
        dat_dir = self.folder_input.text().strip()
        if not dat_dir or not os.path.isdir(dat_dir):
            self.log_message("❌ 오류: 유효한 DAT 파일 폴더를 선택하세요.")
            return
        
        # 설정 값 가져오기
        sensor_type = "ACC" if self.acc_radio.isChecked() else "MIC"
        sampling_rate = self.rate_input.value()
        integration_mode = self.integration_combo.currentData()
        
        # UI 업데이트
        self.convert_button.setEnabled(False)
        self.convert_button.setText("변환 중...")
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # 작업 스레드 시작
        self.worker = ConversionWorker(
            sensor_type, dat_dir, sampling_rate, integration_mode
        )
        self.worker.progress_signal.connect(self.log_message)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished_signal.connect(self.conversion_finished)
        self.worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())