import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os

def load_trained_model(model_path='results/'):
    """훈련된 모델을 로드합니다."""
    if not os.path.exists(model_path):
        print(f"모델 경로 {model_path}를 찾을 수 없습니다.")
        return None, None
    
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        
        # GPU 사용 가능 여부 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"모델이 {device}에 로드되었습니다.")
        return model, tokenizer
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None, None

def generate_response(model, tokenizer, user_input, max_length=100, temperature=0.8):
    """사용자 입력에 대한 응답을 생성합니다."""
    # 대화 형식으로 입력 구성
    prompt = f"상대: {user_input}\n나:"
    
    # 입력 토큰화
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    
    # 패딩 토큰 설정
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    with torch.no_grad():
        # 응답 생성
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 디버깅용 (처음 몇 번만)
    print(f"[DEBUG] 전체 생성 텍스트: {generated_text}")
    
    # "나:" 이후의 부분만 추출하고 구분자 제거
    if "나:" in generated_text:
        response = generated_text.split("나:")[-1].strip()
    else:
        response = generated_text
    
    # 구분자 제거
    response = response.replace("상대:", "").replace("나:", "").strip()
    
    # 여러 줄인 경우 첫 번째 줄만 사용
    if "\n" in response:
        response = response.split("\n")[0].strip()
    
    print(f"[DEBUG] 정리된 응답: {response}")
    
    return response

def interactive_chat():
    """대화형 채팅 인터페이스"""
    model, tokenizer = load_trained_model()
    
    if model is None or tokenizer is None:
        print("모델을 로드할 수 없습니다. 먼저 훈련을 완료해주세요.")
        return
    
    print("채팅을 시작합니다! (종료하려면 'quit' 또는 'exit'를 입력하세요)")
    print("-" * 50)
    
    while True:
        user_input = input("상대방: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("채팅을 종료합니다.")
            break
        
        if not user_input:
            continue
        
        try:
            response = generate_response(model, tokenizer, user_input)
            print(f"나: {response}")
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    interactive_chat() 