from prcwatermark.water_llm import WaterLLM
from prcwatermark.sampler import Sampler
from prcwatermark.prc import PRC
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def sparsity_fn(codeword_len):
    log_base = 3
    return int(np.log2(codeword_len) / np.log2(log_base))

if __name__ == "__main__":
    temperature = 1.4
    repetition_penalty = 1.0
    top_p = 0.8
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = "auto", device_map = "auto")
    sampler = Sampler(model, tokenizer, temperature, repetition_penalty, top_p)


    prc = PRC(63, sparsity_fn)
    llm = WaterLLM(sampler, prc)

    prompt = "You are an educational assistant. Please write a text that is informative and helpful. Write an essay on the use of AI in education."
    is_water = True
    false_positive_rate = 0.01
    response = llm.gen_response(prompt, is_water)
    prob_water = llm.detect_water(response, 0.01)
    print("Probability Watermarked:", prob_water)
    
    



    








