#!/usr/bin/env python
# manual_s3_down.py

import boto3
import os
import shutil
from common_utils import load_aws_config, load_config

def download_manual_files():
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
    
    bucket = "vtnnbl"
    prefix = f"{machine_id}/manual_down/"
    
    print(f"📥 Checking s3://{bucket}/{prefix}")
    
    # 대상 경로 매핑
    path_mapping = {
        f"{prefix}acc/model/": "/home/kks/PDM_RUN/models/acc/current_model/",
        f"{prefix}acc/scaler/": "/home/kks/PDM_RUN/models/acc/current_scaler/",
        f"{prefix}mic/model/": "/home/kks/PDM_RUN/models/mic/current_model/",
        f"{prefix}mic/scaler/": "/home/kks/PDM_RUN/models/mic/current_scaler/"
    }
    
    try:
        # 각 경로별로 파일 확인 및 다운로드
        total_files = []
        file_mappings = {}
        
        for s3_prefix, local_dir in path_mapping.items():
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=s3_prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    if not obj['Key'].endswith('/'):
                        total_files.append(obj['Key'])
                        file_mappings[obj['Key']] = local_dir
        
        if not total_files:
            print("No files found in any path.")
            return
        
        # 파일 목록 표시
        print(f"\n📋 Found {len(total_files)} file(s):")
        print("-" * 80)
        
        for i, file_key in enumerate(total_files):
            filename = os.path.basename(file_key)
            sensor_type = file_key.split('/')[-3]  # acc or mic
            file_type = file_key.split('/')[-2]    # model or scaler
            print(f"  {i+1:2d}. [{sensor_type.upper()}] {file_type:6s} → {filename}")
        
        print("-" * 80)
        
        # 다운로드 확인
        confirm = input("\nDownload all files and deploy to models? (y/n): ")
        if confirm.lower() != 'y':
            print("❌ Download cancelled.")
            return
        
        # 백업 확인
        backup_needed = False
        for local_dir in set(file_mappings.values()):
            if os.path.exists(local_dir) and os.listdir(local_dir):
                backup_needed = True
                break
        
        if backup_needed:
            backup_confirm = input("\n⚠️  기존 파일이 존재합니다. 백업하시겠습니까? (y/n): ")
            create_backup = backup_confirm.lower() == 'y'
        else:
            create_backup = False
        
        # 파일 다운로드 및 배치
        success_count = 0
        
        for file_key in total_files:
            filename = os.path.basename(file_key)
            local_dir = file_mappings[file_key]
            local_path = os.path.join(local_dir, filename)
            
            sensor_type = file_key.split('/')[-3]
            file_type = file_key.split('/')[-2]
            
            print(f"\n📥 Downloading [{sensor_type.upper()}] {file_type}: {filename}")
            
            try:
                # 디렉토리 생성
                os.makedirs(local_dir, exist_ok=True)
                
                # 기존 파일 백업
                if create_backup and os.path.exists(local_path):
                    backup_path = local_path + ".backup"
                    shutil.copy2(local_path, backup_path)
                    print(f"   💾 Backed up existing file")
                
                # S3에서 다운로드
                s3.download_file(bucket, file_key, local_path)
                print(f"   ✅ Downloaded to: {local_path}")
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # 결과 요약
        print("\n" + "=" * 80)
        print(f"📊 Download Summary:")
        print(f"   - Total files: {len(total_files)}")
        print(f"   - Success: {success_count}")
        print(f"   - Failed: {len(total_files) - success_count}")
        
        if success_count == len(total_files):
            print("\n✅ All files downloaded and deployed successfully!")
            print("\n📁 Deployment locations:")
            print("   - ACC Model:   /home/kks/PDM_RUN/models/acc/current_model/")
            print("   - ACC Scaler:  /home/kks/PDM_RUN/models/acc/current_scaler/")
            print("   - MIC Model:   /home/kks/PDM_RUN/models/mic/current_model/")
            print("   - MIC Scaler:  /home/kks/PDM_RUN/models/mic/current_scaler/")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_manual_files()

    '''
    s3://버킷명/{machine_id}/manual_down/
├── acc/
│   ├── model/
│   │   ├── acc_20240101_120000_model.pkl
│   │   └── acc_20240101_120000_model_info.json
│   └── scaler/
│       └── acc_20240101_120000_scaler.pkl
└── mic/
    ├── model/
    │   ├── mic_20240101_120000_model.pkl
    │   └── mic_20240101_120000_model_info.json
    └── scaler/
        └── mic_20240101_120000_scaler.pkl
    
    
    '''