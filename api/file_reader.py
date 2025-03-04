import os

log_directory = 'kakaotalkLog'  # kakaotalkLog 폴더 경로
my_messages = []
other_messages = []
my_name = os.environ['MY_NAME']
def read_log_files():
    try:
        # 폴더 내의 파일 목록을 읽어들임
        files = os.listdir(log_directory)
        print("KakaoTalk Log Files:")
        for file in files:
            print(file)
    except FileNotFoundError:
        print(f"Error: The directory '{log_directory}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return files

def open_file(file_name):
    file_name = log_directory+"/"+file_name
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            contents = file.readlines()  # 행 단위로 읽기
            previous_user = ''
            message = ''
            for line in contents:
                messages = line.strip().split(" ")
                user = messages[0]
                if '[' in user and ']' in user and len(messages) >= 3:
                    if previous_user != user and previous_user == my_name:
                        print(previous_user + " : " + message)
                        my_messages.append(message)
                        message = ''
                    elif previous_user != user and previous_user != my_name:
                        print(previous_user + " : " + message)
                        other_messages.append(message)
                        message = ''
                    for i in range(3, len(messages)):
                        message += messages[i]+ " "
                else :
                    user = ''
                    message = ''
                previous_user = user
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return my_messages, other_messages
