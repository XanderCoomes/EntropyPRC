import torch
import math
import numpy as np
from prcwatermark.subset_sum import calc_subset_sum

class WaterLLM: 
    def __init__(self, sampler, prc, entropy_threshold): 
        self.sampler = sampler
        self.prc = prc
        self.entropy_threshold = entropy_threshold
        self.hash_bitmask = self.simple_hash_bitmask
    
    def hash_from_bitmask(self, token_id, bitmask): 
        return bitmask[token_id]
    
    def simple_hash_bitmask(self, probs): 
        alphabet_size = probs.numel()
        hashes = torch.tensor([i % 2 for i in range(alphabet_size)])
        return hashes

    def ideal_hash_bitmask(self, probs):
        decimals = 4
        device = probs.device 
        non_zero_idx = torch.nonzero(probs, as_tuple = True)[0]
        weights = probs[non_zero_idx].float().cpu().numpy()
        subset_bitmask = calc_subset_sum(weights, 0.5, decimals)
        bitmask = torch.zeros_like(probs, dtype=torch.int)
        bitmask[non_zero_idx] = torch.tensor(subset_bitmask, dtype=torch.int, device=device)
        return bitmask
        
    def gen_response(self, prompt, is_water):
        codeword = self.prc.encode(noise_rate = 0.0)
        generated_ids = self.sampler.txt_to_ids(prompt)
        prompt_tokens = generated_ids.size(1)
        high_entropy_tokens = 0
        key_vals = None
        encoding_errors = 0
        hash_mask = None

        while high_entropy_tokens < len(codeword):
            probs, key_vals = self.sampler.calc_probs(generated_ids, key_vals)
            entropy = self.entropy(probs)
            if(entropy >= self.entropy_threshold):
                if(is_water):
                    hash_mask = self.hash_bitmask(probs) 
                    probs = self.bias_probs(probs, hash_mask, codeword[high_entropy_tokens])
                high_entropy_tokens += 1

            token_id = self.sampler.sample(probs)
            token = self.sampler.tokenizer.decode([token_id], skip_special_tokens = True)
            if(entropy >= self.entropy_threshold and is_water):
                if(self.hash_from_bitmask(token_id, hash_mask).item() != codeword[high_entropy_tokens - 1]): 
                    encoding_errors += 1
                    print(f"\033[41m{token}\033[0m", flush = True, end="") 
                else:
                    print(f"\033[42m{token}\033[0m", flush = True, end="")            
            else:
                print(token, end = '', flush = True)
            generated_ids = torch.cat([generated_ids, torch.tensor([[token_id]])], dim=-1)

        num_tokens_generated = generated_ids.size(1) - prompt_tokens
        encoding_error_rate = encoding_errors / len(codeword)
        print()
        print("-------------------------------------------")
        print("Tokens Generated:", num_tokens_generated)
        print("Encoding Error Rate:", encoding_error_rate)

        return self.sampler.ids_to_txt(generated_ids[0, prompt_tokens:].tolist())
    
    def recover_bit_str(self, response, prompt):
        generated_ids = self.sampler.txt_to_ids(prompt)
        response_ids = self.sampler.txt_to_ids(response)
        response_idx = 0
        high_entropy_tokens = 0
        key_vals = None
        bit_str = [] 
        while high_entropy_tokens < self.prc.codeword_len:
            probs, key_vals = self.sampler.calc_probs(generated_ids, key_vals)
            token_id = response_ids[0, response_idx].item()            
            if(self.entropy(probs) >= self.entropy_threshold):
                hash_mask = self.hash_bitmask(probs)  
                bit_str.append(self.hash_from_bitmask(token_id, hash_mask))
                high_entropy_tokens += 1
            response_idx = response_idx + 1

            generated_ids = torch.cat([generated_ids, torch.tensor([[token_id]])], dim=-1)

        return bit_str

    def detect_water(self, response, prompt): 
        print("-------------------------------------------")
        if(prompt == None): 
            prompt = self.default_prompt
        bit_str = self.recover_bit_str(response, prompt)
        bit_str = np.fromiter(bit_str, dtype = np.uint8, count = len(bit_str))
        print("-------------------------------------------")
        false_positive_rates = [0.0001, 0.001, 0.01,0.02, 0.03,0.04,0.05]
        for fpr in false_positive_rates:
            print(f"False Positive Rate : {fpr}")
            is_water = self.prc.threshold_decode(bit_str, fpr)
            print(f"Watermark Detected  : {is_water}")
            print("-------------------------------------------")
    

    def bias_probs(self, probs, hash_mask, bit): 
        mask = (hash_mask == bit)
        unbiased_prob_sum = probs[mask].sum()
        if unbiased_prob_sum == 0.0 or unbiased_prob_sum == 1.0:
            return probs
        if(unbiased_prob_sum < 0.5):
            biased_prob_sum = 2 * unbiased_prob_sum
        else: 
            biased_prob_sum = 1
        probs[mask] = (probs[mask] / unbiased_prob_sum) * biased_prob_sum
        probs[~mask] = probs[~mask] * ((1 - biased_prob_sum) / (1 - unbiased_prob_sum))

        return probs
        
    def entropy(self, probs): 
        log_probs = torch.log2(probs + 1e-12)
        entropy = -torch.sum(probs * log_probs).item()
        return entropy
    
    