import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments

# KoGPT2 모델과 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

# 패딩 토큰 설정
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 텍스트 파일 읽기
with open("KakaoTalk_20250119_2321_46_139_group.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, tokenizer, text, block_size=512):
        # attention_mask를 포함하여 토큰화
        self.examples = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=block_size, 
            truncation=True, 
            padding="max_length"
        )
        # labels를 input_ids와 동일하게 설정
        self.examples["labels"] = self.examples["input_ids"].clone()
    
    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, i):
        # input_ids, attention_mask, labels 반환
        return {
            # 이쪽이 입력값
            "input_ids": self.examples["input_ids"][i],
            # 이쪽이 마스크
            "attention_mask": self.examples["attention_mask"][i],
            # 이쪽이 라벨 입력값에 대한 결과값
            "labels": self.examples["labels"][i]
        }

# 데이터셋 생성
dataset = TextDataset(tokenizer, text_data)

# 데이터 로더 생성
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 모델 학습
trainer.train()
