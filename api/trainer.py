import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from file_reader import read_log_files, open_file
from transformers import AutoTokenizer

# KoGPT2 모델과 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

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
    def __init__(self, text_data):
        self.target_texts = text_data[0]
        self.input_texts = text_data[1]
        self.length = min(len(self.target_texts), len(self.input_texts))
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of bounds")
            
        target_text = self.target_texts[idx]
        input_text = self.input_texts[idx]
        
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

file_names = read_log_files()
for file in file_names:
    text_data = open_file(file)
    dataset = CustomDataset(text_data)
    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=False
    )

    def get_latest_checkpoint_path(base_path='results'):
        existing_checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]
        checkpoint_numbers = [int(d.split('-')[1]) for d in existing_checkpoints if d.split('-')[1].isdigit()]
        if checkpoint_numbers:
            latest_checkpoint_number = max(checkpoint_numbers)
            return os.path.join(base_path, f'checkpoint-{latest_checkpoint_number}')
        return None

    # latest_checkpoint_path = get_latest_checkpoint_path()
    # if latest_checkpoint_path:
    #     print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
    # else:
    #     print("No checkpoint found, starting training from scratch")

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        fp16=True,
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    try:
        # 학습 시작 (체크포인트에서 재개)
        # model.load_state_dict(torch.load(latest_checkpoint_path), strict=False)
        # trainer.train(resume_from_checkpoint=latest_checkpoint_path)
        trainer.train()

        # 모델과 토크나이저 저장
        model.save_pretrained('results/')
        tokenizer.save_pretrained('results/')

        torch.cuda.empty_cache()
    except Exception as e:
        print(f"An error occurred: {e}")
        torch.cuda.empty_cache()