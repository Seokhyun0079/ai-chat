import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

print("호출: test.py")

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'
sent = '0'

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda사용가능하면 or cpu
model.load_state_dict(torch.load("kogpt2_model_state_dict.pt",map_location=device),strict = False)

def chat(input: str):
    q=input.strip()
    a=""
    while 1:
        input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
        pred = model(input_ids)
        pred = pred.logits
        gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
        if gen == EOS:
            break
        a += gen.replace("▁", " ")
    return a.strip()