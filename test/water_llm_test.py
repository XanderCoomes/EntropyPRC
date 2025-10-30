from prcwatermark.water_llm import WaterLLM
from prcwatermark.sampler import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def sparsity_fn(codeword_len):
    return int(np.log10(codeword_len))

def hash_fn(token_id): 
    return token_id % 2

class WaterLLMTester:
    def __init__(self, llm): 
        self.llm = llm
    
    def test_generation(self, prompt, num_tokens, is_water): 
        response = self.llm.gen_response(prompt, num_tokens, is_water)
        return response
    
    def test_detection(self, response): 
        is_water = self.llm.detect_water(response)
        return is_water


if __name__ == "__main__":
    temperature = 1.8
    repetition_penalty = 1.0
    top_p = 0.8

    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = "auto", device_map = "auto")
    sampler = Sampler(model, tokenizer, temperature, repetition_penalty, top_p)

    key_folder = "./keys"
    llm = WaterLLM(sampler, hash_fn, sparsity_fn, key_folder)
    water_tester = WaterLLMTester(llm)
    response = water_tester.test_generation("You area a helpful, educational assistant. Write a long essay on Abraham Lincoln. ", 100, True)
    is_water = water_tester.test_detection(response)
    if(is_water):
        print("Watermark Detected.")
    else: 
        print("No Watermark Detected.")




    








