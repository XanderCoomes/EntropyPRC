from prcwatermark.water_llm import WaterLLM
from prcwatermark.sampler import Sampler
from prcwatermark.prc import PRC
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def sparsity_fn(codeword_len):
    return int(np.log2(codeword_len))

if __name__ == "__main__":
    temperature = 1.5
    repetition_penalty = 1.0
    top_p = 0.8
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = "auto", device_map = "auto")
    sampler = Sampler(model, tokenizer, temperature, repetition_penalty, top_p)


    prc = PRC(60, sparsity_fn)
    llm = WaterLLM(sampler, prc)

    prompt = "You are an educational assistant. Please write a text that is informative and helpful, explaining the trolley problem in depth."
    is_water = False
    response = llm.gen_response(prompt, is_water)
    prob_water = llm.prob_water(response)
    print("Probability Watermarked:", prob_water)
    
    



    








