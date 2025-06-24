from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
import uvicorn
import os
import torch
from torch.utils.data import Dataset
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

# EOS 토큰이 없으면 추가
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})

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
    max_length: int = 300

def generate_response(prompt, max_length=2000):
    # 입력 문장을 토큰화
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_length = input_ids.shape[1]

    # max_length가 너무 작으면 최소값으로 강제
    if max_length < 50:
        max_length = 50

    # 입력이 너무 길면 에러 반환
    if input_length > max_length - 10:
        return "입력 프롬프트가 너무 깁니다. 더 짧게 입력해 주세요."

    max_new_tokens = max(1, min(max_length - input_length, 300))

    # 모델을 사용하여 응답 생성
    output = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        num_return_sequences=1, 
        no_repeat_ngram_size=3,  # 더 큰 n-gram으로 반복 방지
        do_sample=True,  # 확률적 샘플링 활성화
        temperature=0.9,  # 창의성 증가 (0.7 -> 0.9)
        top_p=0.85,  # nucleus sampling 조정
        top_k=50,  # top-k 샘플링 추가
        early_stopping=True,  # 자연스러운 종료 지점에서 멈춤
        pad_token_id=tokenizer.eos_token_id,  # EOS 토큰으로 패딩
        eos_token_id=tokenizer.eos_token_id,  # EOS 토큰 설정
        repetition_penalty=1.2,  # 반복 방지 강화
        length_penalty=1.0,  # 길이 페널티
        bad_words_ids=[[tokenizer.encode(prompt, add_special_tokens=False)[0]]] if len(tokenizer.encode(prompt, add_special_tokens=False)) > 0 else None  # 입력의 첫 토큰을 나쁜 단어로 설정
    )
    
    # 생성된 응답을 디코딩
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 더 정확한 프롬프트 제거 로직
    # 입력과 응답을 토큰 단위로 비교하여 정확히 제거
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    response_tokens = tokenizer.encode(response, add_special_tokens=False)
    
    # 입력 토큰이 응답 시작 부분에 있는지 확인하고 제거
    if len(response_tokens) >= len(prompt_tokens) and response_tokens[:len(prompt_tokens)] == prompt_tokens:
        response_tokens = response_tokens[len(prompt_tokens):]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # 여전히 입력이 포함되어 있다면 문자열 레벨에서 제거 시도
    if prompt.strip() in response:
        response = response.replace(prompt.strip(), "").strip()
    
    # 응답이 너무 짧거나 입력과 너무 비슷하면 기본 메시지 반환
    if len(response.strip()) < 5 or response.strip() == prompt.strip():
        response = "죄송합니다. 적절한 응답을 생성하지 못했습니다."
    
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
    print(request.max_length)
    
    # 더 나은 프롬프트 엔지니어링
    enhanced_prompt = request.prompt
    
    response = generate_response(enhanced_prompt, request.max_length)
    return {"response": response}

# 서버 생존 확인 엔드포인트
@app.get("/health/")
async def health_check():
    return {"status": "alive"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
