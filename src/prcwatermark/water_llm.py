import torch
import math
import numpy as np

class WaterLLM: 
    def __init__(self, sampler, prc): 
        self.sampler = sampler
        self.mini_sampler = sampler
        self.prc = prc

        self.default_prompt = "You are an educational assistant. Please write a text that is informative and helpful on free will"
        self.entropy_threshold = 0.8
        self.startup_tokens = 100
    
    def hash_fn(self, token_id): 
        return token_id % 2
    
    def gen_response(self, prompt, is_water):
        codeword = self.prc.encode(noise_rate = 0.0)
        main_generated_ids = self.sampler.txt_to_ids(prompt)
        mini_generated_ids = self.mini_sampler.txt_to_ids(self.default_prompt)
        prompt_tokens = main_generated_ids.size(1)
        high_entropy_tokens = 0
        main_key_vals = None
        mini_key_vals = None
        encoding_errors = 0
        while high_entropy_tokens < len(codeword):
            main_probs, main_key_vals = self.sampler.calc_probs(main_generated_ids, main_key_vals)
            mini_probs, mini_key_vals = self.mini_sampler.calc_probs(mini_generated_ids, mini_key_vals)
            if(self.token_hash_entropy(mini_probs) >= self.entropy_threshold and mini_generated_ids.size(1) > self.startup_tokens):
                if(is_water): 
                    main_probs = self.bias_probs(main_probs, codeword[high_entropy_tokens])
                high_entropy_tokens += 1
        
            token_id = self.sampler.sample(main_probs)
            token = self.sampler.tokenizer.decode([token_id], skip_special_tokens = True)
            if(self.token_hash_entropy(mini_probs) >= self.entropy_threshold and self.hash_fn(token_id) != codeword[high_entropy_tokens - 1] and mini_generated_ids.size(1) > self.startup_tokens): 
                encoding_errors += 1
            print(token, end = '', flush = True)
            main_generated_ids = torch.cat([main_generated_ids, torch.tensor([[token_id]])], dim=-1)
            mini_generated_ids = torch.cat([mini_generated_ids, torch.tensor([[token_id]])], dim=-1)
        
        print("\n\n")
        response = self.sampler.ids_to_txt(main_generated_ids[0, prompt_tokens:].tolist())
        print(f"Encoding Error Rate: {encoding_errors / len(codeword)}")

        return response

    def high_entropy_mask(self, response): 
        mini_generated_ids = self.mini_sampler.txt_to_ids(self.default_prompt)
        main_generated_ids = self.sampler.txt_to_ids(response)
        mask = torch.zeros(main_generated_ids.size(1), dtype = torch.bool)
        mini_key_vals = None
        for i in range(main_generated_ids.size(1)):
            mini_probs, mini_key_vals = self.mini_sampler.calc_probs(mini_generated_ids, mini_key_vals)
            token_id = main_generated_ids[0, i].item()
            if(self.token_hash_entropy(mini_probs) >= self.entropy_threshold and mini_generated_ids.size(1) > self.startup_tokens):
                mask[i] = True
            mini_generated_ids = torch.cat([mini_generated_ids, torch.tensor([[token_id]])], dim=-1)
        return mask
    
    def recover_bit_str(self, response):
        mask = self.high_entropy_mask(response) 
        bits = []
        main_generated_ids = self.sampler.txt_to_ids(response)
        for i in range(main_generated_ids.size(1)):
            if(mask[i]):
                token_id = main_generated_ids[0, i].item()
                bits.append(self.hash_fn(token_id))

        return bits
    

    def prob_water(self, response): 
        bit_str = self.recover_bit_str(response)
        bit_str = np.fromiter(bit_str, dtype = np.uint8, count = len(bit_str))
        noise_rate = self.calc_approx_error_rate()
        return self.prc.prob_codeword(bit_str, noise_rate)
        
    def binary_entropy(self, p):
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    def token_hash_entropy(self, probs): 
        alphabet_size = probs.numel()
        hashes = torch.tensor([self.hash_fn(i) for i in range(alphabet_size)])
        mask = (hashes == 0)
        p = probs[mask].sum().item()
        entropy = self.binary_entropy(p)
        return entropy
    
    def calc_approx_error_rate(self): 
        p = self.p_given_entropy(self.entropy_threshold)
        beta = 0.5 - p
        return beta / 2

    def bias_probs(self, probs, bit): 
        alphabet_size = probs.numel()
        hashes = torch.tensor([self.hash_fn(i) for i in range(alphabet_size)])
        mask = (hashes == bit)
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
    
    def p_given_entropy(self, entropy): 
        tol = 10e-12
        max_iter = 1000
        lower = 0.0
        upper = 0.5
        for _ in range(max_iter):
            mid = (lower + upper) / 2
            h = self.binary_entropy(mid)
            if abs(h - entropy) < tol:
                return mid
            if h < entropy:
                lower = mid
            else:
                upper = mid
        return (lower + upper) / 2

        
       
    