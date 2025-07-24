import os
import re

def rename_files(directory):
    # 디렉토리 내의 모든 파일 검색
    for filename in os.listdir(directory):
        # 파일 이름 패턴 매칭 (MIC와 ACC 모두 처리)
        pattern = r'\d{2}_(\d{8})_(\d{2})_(\d{2})_(\d{2})_(MP\d+ABS\d+_MIC|LSM6DSOX_ACC)\.dat'
        match = re.match(pattern, filename)
        
        if match:
            # 패턴에서 필요한 정보 추출
            date = match.group(1)
            hour = match.group(2)
            minute = match.group(3)
            second = match.group(4)
            sensor_id = match.group(5)
            
            # 새로운 파일 이름 생성
            new_filename = f"{sensor_id}_{date}_{hour}_{minute}_{second}.dat"
            
            # 전체 경로 생성
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            try:
                # 파일 이름 변경
                os.rename(old_path, new_path)
                print(f"변경 완료: {filename} -> {new_filename}")
            except Exception as e:
                print(f"오류 발생: {filename} - {str(e)}")

if __name__ == "__main__":
    # 사용자로부터 디렉토리 경로 입력 받기
    directory = input("파일이 있는 디렉토리 경로를 입력하세요: ")
    
    # 디렉토리가 존재하는지 확인
    if os.path.exists(directory):
        rename_files(directory)
        print("모든 파일 이름 변경이 완료되었습니다.")
    else:
        print("입력한 디렉토리가 존재하지 않습니다.")