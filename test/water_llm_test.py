from prcwatermark.water_llm import WaterLLM
from prcwatermark.sampler import Sampler
from prcwatermark.prc import PRC
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def sparsity_fn(codeword_len):
    log_base = 2
    return int(np.log2(codeword_len) / np.log2(log_base))

if __name__ == "__main__":
    temperature = 1.4
    repetition_penalty = 1.0
    top_p = 0.8
    codeword_len = 64
    
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = "auto", device_map = "auto")
    sampler = Sampler(model, tokenizer, temperature, repetition_penalty, top_p)
    prc = PRC(codeword_len, sparsity_fn)
   
    startup_tokens = 0
    entropy_threshold = 0.8

    prompt = "You are an educational assistant. Please write a text that is informative and helpful. Write an essay about playing pool."
    is_water = True
    llm = WaterLLM(sampler, prc, entropy_threshold, startup_tokens)
    response = llm.gen_response(prompt, is_water)
    llm.detect_water(response, prompt)

    
    



    








