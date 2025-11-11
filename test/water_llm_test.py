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
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = "auto", device_map = "auto")
    sampler = Sampler(model, tokenizer, temperature, repetition_penalty, top_p)


    codeword_len = 64
    prc = PRC(codeword_len, sparsity_fn)
    startup_tokens = 150
    entropy_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    trials_per_threshold = 10

    prompt = "You are an educational assistant. Please write a text that is informative and helpful. Write an essay on the use of AI in education."
    is_water = True
    avg_tokens_generated = np.zeros_like(entropy_thresholds, dtype = float)
    avg_encoding_error_rate = np.zeros_like(entropy_thresholds, dtype = float)
    for i, entropy_threshold in enumerate(entropy_thresholds):
        print(f"Testing Entropy Threshold: {entropy_threshold}")
        for j in range(trials_per_threshold):
            print(f" Trial {j + 1}/{trials_per_threshold}")
            llm = WaterLLM(sampler, prc, entropy_threshold, startup_tokens)
            response, encoding_error_rate, num_tokens = llm.gen_response(prompt, is_water)
            avg_tokens_generated[i] += num_tokens - startup_tokens
            avg_encoding_error_rate[i] += encoding_error_rate
        avg_tokens_generated[i] /= trials_per_threshold
        avg_encoding_error_rate[i] /= trials_per_threshold
    
    avg_token_expansion_rate = avg_tokens_generated / codeword_len
    
    print("Entropy Thresholds: ", entropy_thresholds)
    print("Average Tokens Expansion Rate: ", avg_token_expansion_rate)
    print("Average Encoding Error Rates: ", avg_encoding_error_rate)
    
    



    








