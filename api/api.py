from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import uvicorn

# FastAPI 앱 생성
app = FastAPI()

# KoGPT2 모델과 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

# 요청 바디 모델 정의
class ChatRequest(BaseModel):
    prompt: str
    max_length: int = 50

def generate_response(prompt, max_length=200):
    # 입력 문장을 토큰화
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # 모델을 사용하여 응답 생성
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # 생성된 응답을 디코딩
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# 챗봇 응답 엔드포인트
@app.post("/chat/")
async def chat(request: ChatRequest):
    response = generate_response(request.prompt, request.max_length)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
