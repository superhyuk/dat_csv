#!/usr/bin/env python
# manual_s3_up.py
#라즈베리파이에서 업로드 하기 위한 파일일
import boto3
import os
from common_utils import load_aws_config, load_config

def upload_current_models():
    # AWS 설정
    aws_cfg = load_aws_config()
    s3 = boto3.client('s3',
        aws_access_key_id=aws_cfg["access_key"],
        aws_secret_access_key=aws_cfg["secret_key"],
        region_name=aws_cfg["region"]
    )
    
    # 현재 머신 ID
    config = load_config()
    machine_id = config.get("machine_id")
    bucket = aws_cfg["bucket_name"]
    
    # 업로드할 폴더들
    base_path = "/home/kks/PDM_RUN/models"
    sensors = ["mic", "acc"]
    folders = ["current_model", "current_scaler"]
    
    # 업로드할 파일 목록 수집
    files_to_upload = []
    
    for sensor in sensors:
        for folder in folders:
            folder_path = os.path.join(base_path, sensor, folder)
            
            if not os.path.exists(folder_path):
                print(f"⚠️  폴더가 존재하지 않음: {folder_path}")
                continue
                
            # 폴더 내 파일들 확인
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    # 상대 경로 계산
                    relative_path = os.path.relpath(local_file, base_path)
                    s3_key = f"{machine_id}/manual_upload/{relative_path}"
                    
                    files_to_upload.append({
                        'local': local_file,
                        's3_key': s3_key,
                        'size': os.path.getsize(local_file)
                    })
    
    if not files_to_upload:
        print("❌ 업로드할 파일이 없습니다.")
        return
    
    # 업로드할 파일 목록 표시
    print(f"\n📤 업로드할 파일 목록 ({len(files_to_upload)}개):")
    print("-" * 80)
    total_size = 0
    
    for idx, file_info in enumerate(files_to_upload, 1):
        size_mb = file_info['size'] / (1024 * 1024)
        print(f"{idx:3d}. {os.path.basename(file_info['local']):30s} ({size_mb:6.2f} MB)")
        print(f"     → s3://{bucket}/{file_info['s3_key']}")
        total_size += file_info['size']
    
    print("-" * 80)
    print(f"총 크기: {total_size / (1024 * 1024):.2f} MB")
    
    # 업로드 확인
    confirm = input("\n모든 파일을 업로드하시겠습니까? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ 업로드가 취소되었습니다.")
        return
    
    # 파일 업로드
    print(f"\n🚀 S3 업로드 시작...")
    success_count = 0
    failed_files = []
    
    for idx, file_info in enumerate(files_to_upload, 1):
        try:
            print(f"\n[{idx}/{len(files_to_upload)}] 업로드 중: {os.path.basename(file_info['local'])}")
            print(f"  크기: {file_info['size'] / (1024 * 1024):.2f} MB")
            print(f"  대상: s3://{bucket}/{file_info['s3_key']}")
            
            # S3 업로드
            s3.upload_file(
                file_info['local'],
                bucket,
                file_info['s3_key']
            )
            
            print(f"  ✅ 업로드 완료!")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 업로드 실패: {e}")
            failed_files.append(file_info['local'])
    
    # 결과 요약
    print("\n" + "=" * 80)
    print(f"📊 업로드 결과:")
    print(f"  - 성공: {success_count}/{len(files_to_upload)}")
    print(f"  - 실패: {len(failed_files)}")
    
    if failed_files:
        print(f"\n❌ 실패한 파일:")
        for file in failed_files:
            print(f"  - {file}")
    else:
        print(f"\n✅ 모든 파일이 성공적으로 업로드되었습니다!")
        print(f"   S3 경로: s3://{bucket}/{machine_id}/manual_upload/")

if __name__ == "__main__":
    try:
        upload_current_models()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")