from prcwatermark.prc import PRC
from pathlib import Path
import numpy as np
import pickle
import torch
import math

class WaterLLM: 
    def __init__(self, sampler, hash_fn, sparsity_fn, entropy_threshold, key_folder): 
        self.sampler = sampler
        self.hash_fn = hash_fn
        self.sparsity_fn = sparsity_fn
        self.entropy_threshold = entropy_threshold
        self.key_folder = key_folder

    def sample_codeword(self, codeword_len):
        prc = self.load_prc(codeword_len)
        codeword = prc.encode(noise_rate = 0.0)
        return codeword

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
    
    def print_token(self, token, color):
        if(color == 'green'):
            print(f"\033[42m{token}\033[0m", end = '', flush = True)
        elif(color == 'red'):
            print(f"\033[41m{token}\033[0m", end = '', flush = True)
        else:
            print(token, end = '', flush = True)
    
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
    
    def gen_response(self, prompt, codeword_len, is_water):
        codeword = self.sample_codeword(codeword_len)
        generated_ids = self.sampler.text_to_ids(prompt)
        prompt_tokens = generated_ids.size(1)
        past_key_values = None
        high_entropy_tokens = 0
        high_entropy_positions = []
        while high_entropy_tokens < codeword_len:
            probs, past_key_values = self.sampler.calc_probs(generated_ids, past_key_values)
            if(self.token_hash_entropy(probs) >= self.entropy_threshold):
                high_entropy_positions.append(True)
                if(is_water):
                    probs = self.bias_probs(probs, codeword[high_entropy_tokens])
                high_entropy_tokens += 1
            else:
                high_entropy_positions.append(False)
            token_id = self.sampler.sample(probs)
            token = self.sampler.tokenizer.decode([token_id], skip_special_tokens = True)
            self.print_token(token, color = None)
            generated_ids = torch.cat([generated_ids, torch.tensor([[token_id]])], dim=-1)
            
        print("\n\nStatistics:")
        return generated_ids[0, prompt_tokens:].tolist(), high_entropy_positions
    
    def high_entropy_ids(self, generated_ids, high_entropy_positions):
        high_entropy_ids = []
        for i in range(len(generated_ids)):
            if(high_entropy_positions[i]):
                high_entropy_ids.append(generated_ids[i])
        return high_entropy_ids
    
    def p_given_entropy(self, entropy, tol = 1e-12, max_iter = 1000): 
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
    
    def calc_approx_error_rate(self): 
        p = self.p_given_entropy(self.entropy_threshold)
        beta = 0.5 - p
        return beta / 2

    def detect_water(self, generated_ids, high_entropy_positions):
        high_entropy_ids = self.high_entropy_ids(generated_ids, high_entropy_positions) 
        codeword_len = len(high_entropy_ids)
        if(self.prc_exists(codeword_len) == False):
            print(f"PRC of Codeword Length {codeword_len} does not exist.")
            return False
        bit_str = [self.hash_fn(high_entropy_ids[i]) for i in range(codeword_len)]
        bit_str = np.fromiter(bit_str, dtype = np.uint8, count = codeword_len)
        prc = self.load_prc(codeword_len)
        noise_rate = self.calc_approx_error_rate()
        return prc.prob_codeword(bit_str, noise_rate)

    def prc_file_name(self, codeword_len):
        return str(codeword_len) + "bit_prc.pkl"
    
    def save_prc(self, prc):
        folder = Path(self.key_folder)
        folder.mkdir(parents = True, exist_ok = True)
        with open(folder / self.prc_file_name(prc.codeword_len), "wb") as f:
            pickle.dump(prc, f, protocol = pickle.HIGHEST_PROTOCOL)

    def prc_exists(self, codeword_len):
        file_name = self.prc_file_name(codeword_len)
        file_path = Path(self.key_folder) / file_name
        return file_path.exists()
    
    def load_prc(self, codeword_len): 
        file_name = self.prc_file_name(codeword_len)
        file_path = Path(self.key_folder) / file_name
        if(self.prc_exists(codeword_len)):
            with open(Path(file_path), "rb") as f:
                prc =  pickle.load(f)
        else:
            prc = PRC(codeword_len, self.sparsity_fn)
            self.save_prc(prc)
        return prc
        

    