% === 각 CSV 파일을 개별 변수로 저장하는 매크로 ===

% 1. CSV 파일이 있는 폴더 지정
folder = 'D:\MICDATA\rad_da_py\CURING\downloaded_data\acc\2025-05-04\csv'; % <- 본인 폴더 경로 입력

% 2. 폴더 존재 여부 확인
if ~exist(folder, 'dir')
    error('폴더가 존재하지 않습니다: %s', folder);
end

% 3. 모든 파일 확인
all_files = dir(folder);
fprintf('폴더 내 모든 파일:\n');
for i = 1:length(all_files)
    if ~all_files(i).isdir
        fprintf('%s\n', all_files(i).name);
    end
end

% 4. CSV 파일만 필터링
files = dir(fullfile(folder, '*.csv'));
num_files = length(files);

if num_files == 0
    error('CSV 파일을 찾을 수 없습니다. 파일 확장자가 .csv인지 확인해주세요.');
end

% 5. CSV 파일 목록 출력
fprintf('\nCSV 파일 목록:\n');
for i = 1:num_files
    fprintf('%s\n', files(i).name);
end

% 6. 각 파일을 개별 변수로 저장
for k = 1:num_files
    % 파일 이름에서 확장자 제거
    [~, filename, ~] = fileparts(files(k).name);
    
    % 파일 이름을 유효한 변수명으로 변환
    % (공백, 특수문자 등을 언더스코어로 변경)
    varname = regexprep(filename, '[^a-zA-Z0-9]', '_');
    
    % 파일 읽기
    data = readmatrix(fullfile(folder, files(k).name));
    
    % 변수 생성 및 데이터 저장
    eval([varname ' = data;']);
    
    % 처리 결과 출력
    fprintf('파일 %s가 변수 %s로 저장되었습니다.\n', files(k).name, varname);
end

% 7. 작업 공간의 변수 목록 출력
fprintf('\n생성된 변수 목록:\n');
whos('-regexp', '^[a-zA-Z]')