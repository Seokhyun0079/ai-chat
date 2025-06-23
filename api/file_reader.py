import os
import re
from datetime import datetime

log_directory = 'kakaotalkLog'  # kakaotalkLog 폴더 경로
my_name = os.environ.get('MY_NAME', '[황석현]')  # 환경변수가 없으면 기본값 사용

def read_log_files():
    files = []  # 변수를 함수 시작 시 초기화
    try:
        # 폴더 내의 파일 목록을 읽어들임
        files = os.listdir(log_directory)
        print("KakaoTalk Log Files:")
        for file in files:
            print(file)
    except FileNotFoundError:
        print(f"Error: The directory '{log_directory}' does not exist.")
        print("Please create the 'kakaotalkLog' directory and add your KakaoTalk log files.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return files

def clean_message(message):
    """메시지에서 시간 정보와 불필요한 문자를 제거합니다."""
    original_message = message
    
    # [시간] 형식 제거 (예: [3:51], [오후 2:30] 등)
    message = re.sub(r'\[\d{1,2}:\d{2}\]', '', message)
    message = re.sub(r'\[오전 \d{1,2}:\d{2}\]', '', message)
    message = re.sub(r'\[오후 \d{1,2}:\d{2}\]', '', message)
    
    # 시간] 형식 제거 (예: 6:36], 9:10] 등)
    message = re.sub(r'\d{1,2}:\d{2}\]', '', message)
    
    # 앞뒤 공백 제거
    message = message.strip()
    
    # # 디버깅용 출력 (처음 몇 개만)
    # if len(original_message) != len(message):
    #     print(f"정리 전: {original_message}")
    #     print(f"정리 후: {message}")
    #     print("-" * 30)
    
    return message

def open_file(file_name):
    file_name = log_directory+"/"+file_name
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            
        # 대화 메시지를 시간순으로 정렬하기 위한 리스트
        conversation_entries = []
        
        for line in contents:
            line = line.strip()
            if not line:
                continue
                
            # 메시지 파싱
            parts = line.split(" ", 2)  # 최대 2번만 분할
            if len(parts) >= 3:
                user = parts[0]
                time_str = parts[1] if len(parts) > 1 else ""
                message = parts[2] if len(parts) > 2 else ""
                
                # 유효한 사용자명인지 확인
                if '[' in user and ']' in user and message:
                    # 메시지에서 시간 정보 제거
                    clean_msg = clean_message(message)
                    
                    if clean_msg:  # 빈 메시지가 아닌 경우만 추가
                        conversation_entries.append({
                            'user': user,
                            'message': clean_msg,
                            'timestamp': datetime.now()  # 임시로 현재 시간 사용
                        })
        
        # 대화 시퀀스 생성
        conversation_sequences = []
        current_sequence = []
        
        for entry in conversation_entries:
            current_sequence.append(entry)
            
            # 시퀀스가 너무 길어지면 분할 (예: 10개 메시지마다)
            if len(current_sequence) >= 10:
                conversation_sequences.append(current_sequence)
                current_sequence = []
        
        # 마지막 시퀀스 추가
        if current_sequence:
            conversation_sequences.append(current_sequence)
        
        # 대화 형식으로 변환
        formatted_conversations = []
        for sequence in conversation_sequences:
            conversation_text = ""
            for entry in sequence:
                if entry['user'] == my_name:
                    conversation_text += f"나: {entry['message']}\n"
                else:
                    conversation_text += f"상대: {entry['message']}\n"
            formatted_conversations.append(conversation_text.strip())
        
        return formatted_conversations
        
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
