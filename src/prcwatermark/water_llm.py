import torch
import math

class WaterLLM: 
    def __init__(self, sampler, prc): 
        self.sampler = sampler
        self.mini_sampler = sampler
        self.prc = prc

        self.default_prompt = "You are an educational assistant. Please write a text that is informative and helpful."
        self.entropy_threshold = 0.5
    
    def hash_fn(self, token_id): 
        return token_id % 2
    
    def gen_response(self, prompt, is_water):
        codeword = self.prc.encode(noise_rate = 0.0)
        main_generated_ids = self.sampler.text_to_ids(prompt)
        mini_generated_ids = self.mini_sampler.text_to_ids(self.default_prompt)
        prompt_tokens = main_generated_ids.size(1)
        main_key_vals, mini_key_vals = None

        high_entropy_tokens = 0
        while len(high_entropy_tokens) < len(codeword):
            main_probs, main_key_vals = self.sampler.calc_probs(main_generated_ids, main_key_vals)
            mini_probs, mini_key_vals = self.mini_model.calc_probs(mini_generated_ids, mini_key_vals)
            if(self.token_hash_entropy(mini_probs) >= self.entropy_threshold):
                if(is_water): 
                    main_probs = self.bias_probs(main_probs, codeword[high_entropy_tokens])
                high_entropy_tokens += 1
        
            token_id = self.sampler.sample(main_probs)
            token = self.sampler.tokenizer.decode([token_id], skip_special_tokens = True)
            print(token)
            main_generated_ids = torch.cat([main_generated_ids, torch.tensor([[token_id]])], dim=-1)
            mini_generated_ids = torch.cat([mini_generated_ids, torch.tensor([[token_id]])], dim=-1)
        
        response = self.sampler.ids_to_text(main_generated_ids[0, prompt_tokens:].tolist())
        return response


    def prob_water(self, response): 
        pass

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

        
       
    