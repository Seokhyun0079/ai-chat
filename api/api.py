from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
import uvicorn
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from model_selector import SKT_MODEL

trained_model_path = SKT_MODEL

# 모델 설정 로드
config = GPT2Config.from_pretrained(trained_model_path)

# 설정 출력 (필요한 경우 설정을 수정할 수 있습니다)
print(config)

# 수정된 설정으로 모델 로드
model = GPT2LMHeadModel.from_pretrained(trained_model_path, config=config)
tokenizer = GPT2TokenizerFast.from_pretrained(trained_model_path)

# 패딩 토큰 설정
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 모델의 임베딩 레이어 크기 조정
model.resize_token_embeddings(len(tokenizer))

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델을 GPU로 이동
model.to(device)

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, message, response):
        self.target_texts = message
        self.input_texts = response
        self.length = min(len(self.target_texts), len(self.input_texts))
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of bounds")
            
        target_text = self.target_texts
        input_text = self.input_texts
        
        input_encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = input_encoding['input_ids'].flatten()
        target_ids = target_encoding['input_ids'].flatten()
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels
        }

app = FastAPI()


# trainer.py에서 생성한 모델 로드
# 요청 바디 모델 정의
class ChatRequest(BaseModel):
    prompt: str
    previous_message: str
    max_length: int = 300

def generate_response(prompt, max_length=2000):
    # 입력 문장을 토큰화
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  # GPU로 이동
    
    # 모델을 사용하여 응답 생성
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # 생성된 응답을 디코딩
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
def get_latest_checkpoint_path(base_path='results'):
    existing_checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]
    checkpoint_numbers = [int(d.split('-')[1]) for d in existing_checkpoints if d.split('-')[1].isdigit()]
    if checkpoint_numbers:
        latest_checkpoint_number = max(checkpoint_numbers)
        return os.path.join(base_path, f'checkpoint-{latest_checkpoint_number}')
    return None
# 챗봇 응답 엔드포인트
@app.post("/chat/")
async def chat(request: ChatRequest):
    print(request)
    print(request.previous_message)
    print(request.max_length)
    response = generate_response(request.prompt, 200)
    return {"response": response}

# 서버 생존 확인 엔드포인트
@app.get("/health/")
async def health_check():
    return {"status": "alive"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
