from transformers import AutoConfig, GPT2LMHeadModel, GPT2TokenizerFast,  AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
SKT_MODEL = "skt/kogpt2-base-v2"
PHI_MODEL = "microsoft/Phi-4-mini-instruct"
class ModelSelector:
  def __init__(self, model_name):    
    self.model_name = model_name
    self.select_model()
  def select_model(self):
    if self.model_name ==  SKT_MODEL:
      self.model = GPT2LMHeadModel.from_pretrained(SKT_MODEL)
      self.tokenizer = GPT2TokenizerFast.from_pretrained(SKT_MODEL)
      self.batch_size = 4
      self.max_length = 512
      self.fp16 = True
    elif self.model_name == PHI_MODEL:
      self.model = AutoModelForCausalLM.from_pretrained(PHI_MODEL, 
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="offload",
      )
      self.model.eval()
      self.tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL)
      self.batch_size = 1
      self.max_length = 1
      self.fp16 = False
    else:
      raise ValueError("Invalid model name")
  def get_model(self):
    return self.model
  def get_tokenizer(self):
    return self.tokenizer