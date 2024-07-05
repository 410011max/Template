import os
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datautils import *
from eval import *

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' 
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    model_name = "taide/TAIDE-LX-7B-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.seqlen = 2048

    dataloader, testloader = get_loaders(
        'wikitext2', seed=42, model=model_name, seqlen=2048
    )
    print(f"Evaluating {'wikitext2'} ...")
    ppl = llama_eval(model, testloader, 'cuda')
    print(f"targetResult,{'wikitext2'},{ppl:.3f}")