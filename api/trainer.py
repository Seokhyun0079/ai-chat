import os
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from file_reader import read_log_files, open_file
import re
from model_selector import ModelSelector

user_input = input("Select Model: 1.skt/kogpt2-base-v2, 2. Phi-4-mini-instruct-v2 : ")
# KoGPT2 모델과 토크나이저 로드
if user_input == "1":
    model_name = "skt/kogpt2-base-v2"
elif user_input == "2":
    model_name = "microsoft/Phi-4-mini-instruct"
else:
    print("Invalid input")
    exit()

model_selector = ModelSelector(model_name)
model = model_selector.get_model()
tokenizer = model_selector.get_tokenizer()
# 패딩 토큰 설정
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 모델의 임베딩 레이어 크기 조정
model.resize_token_embeddings(len(tokenizer))

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델을 GPU로 이동
model.to(device)

def create_conversation_pairs(my_messages, other_messages):
    """대화 쌍을 만들어서 맥락을 유지하면서 학습할 수 있도록 합니다."""
    conversation_pairs = []
    
    # 메시지들을 시간순으로 정렬 (실제로는 로그 파일의 순서대로)
    min_len = min(len(my_messages), len(other_messages))
    
    for i in range(min_len):
        # 상대방 메시지가 내 메시지보다 먼저 온 경우
        if i < len(other_messages):
            context = other_messages[i]
            response = my_messages[i] if i < len(my_messages) else ""
            
            # 대화 형식으로 만들기
            conversation = f"상대: {context}\n나: {response}"
            conversation_pairs.append(conversation)
    
    return conversation_pairs

def create_training_data(conversation_pairs, max_length=512):
    """대화 데이터를 학습에 적합한 형태로 변환합니다."""
    training_data = []
    
    for conversation in conversation_pairs:
        # 대화를 토큰화
        encoding = tokenizer(
            conversation,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 입력과 레이블을 동일하게 설정 (자기회귀 학습)
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # 레이블은 입력과 동일하되, 패딩 토큰은 -100으로 설정
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        training_data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })
    
    return training_data

# 데이터셋 클래스 정의
class ConversationDataset(Dataset):
    def __init__(self, training_data):
        self.data = training_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 파일들을 읽어서 대화 데이터 생성
file_names = read_log_files()
all_conversations = []

for file in file_names:
    conversations = open_file(file)
    all_conversations.extend(conversations)

print(f"총 {len(all_conversations)}개의 대화 시퀀스를 생성했습니다.")

# 학습 데이터 생성
training_data = create_training_data(all_conversations, model_selector.max_length)
dataset = ConversationDataset(training_data)

print(f"데이터셋 크기: {len(dataset)}")

# 샘플 데이터 출력 (디버깅용)
if len(dataset) > 0:
    sample = dataset[0]
    sample_text = tokenizer.decode(sample['input_ids'])
    print(f"샘플 대화:\n{sample_text[:200]}...")
    print(f"토큰 수: {len(sample['input_ids'])}")

def get_latest_checkpoint_path(base_path=f'./{model_name}'):
    existing_checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]
    checkpoint_numbers = [int(d.split('-')[1]) for d in existing_checkpoints if d.split('-')[1].isdigit()]
    if checkpoint_numbers:
        latest_checkpoint_number = max(checkpoint_numbers)
        return os.path.join(base_path, f'checkpoint-{latest_checkpoint_number}')
    return None

# 학습 설정 개선
training_args = TrainingArguments(
    output_dir=f"./{model_name}/",
    overwrite_output_dir=True,
    num_train_epochs=5,  # 에포크 수 증가
    per_device_train_batch_size=model_selector.batch_size,  # 배치 크기 조정
    save_steps=500,  # 더 자주 저장
    save_total_limit=3,  # 체크포인트 수 증가
    fp16=True,
    logging_steps=100,  # 더 자주 로깅
    learning_rate=5e-5,  # 학습률 조정
    warmup_steps=100,  # 워밍업 스텝 추가
    gradient_accumulation_steps=2,  # 그래디언트 누적
    evaluation_strategy="no",  # 평가 비활성화 (데이터가 적을 수 있으므로)
    load_best_model_at_end=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

try:
    print("훈련을 시작합니다...")
    trainer.train()

    # 모델과 토크나이저 저장
    model.save_pretrained(f'{model_name}/')
    tokenizer.save_pretrained(f'{model_name}/')
    
    print("훈련이 완료되었습니다!")
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"훈련 중 오류가 발생했습니다: {e}")
    
    # 에러 발생 시 해당 모델 폴더 삭제
    model_folder = f'./{model_name}/'
    if os.path.exists(model_folder):
        try:
            shutil.rmtree(model_folder)
            print(f"에러로 인해 {model_folder} 폴더를 삭제했습니다.")
        except Exception as delete_error:
            print(f"폴더 삭제 중 오류가 발생했습니다: {delete_error}")
    
    torch.cuda.empty_cache()