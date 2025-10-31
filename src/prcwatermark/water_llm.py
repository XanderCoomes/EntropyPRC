from prcwatermark.prc import PRC
from pathlib import Path
import numpy as np
import pickle
import torch

class WaterLLM: 
    def __init__(self, sampler, hash_fn, sparsity_fn, key_folder): 
        self.sampler = sampler
        self.hash_fn = hash_fn
        self.sparsity_fn = sparsity_fn
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

    def token_hash_entropy(self, probs): 
        alphabet_size = probs.numel()
        hashes = torch.tensor([self.hash_fn(i) for i in range(alphabet_size)])
        mask = (hashes == 0)
        p = probs[mask].sum()
        entropy = p * torch.log2(p) + (1 - p) * torch.log2(1 - p)
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
            if(self.token_hash_entropy(probs) >= 0):
                high_entropy_tokens += 1
                high_entropy_positions.append(True)
                if(is_water):
                    probs = self.bias_probs(probs, codeword[high_entropy_tokens])
            else:
                high_entropy_positions.append(False)
            token_id = self.sampler.sample(probs)
            token = self.sampler.tokenizer.decode([token_id], skip_special_tokens = True)
            generated_ids = torch.cat([generated_ids, torch.tensor([[token_id]])], dim=-1)
            print(token)
        print("\n\nStatistics:")
        return generated_ids[0, prompt_tokens:].tolist(), high_entropy_positions

    def detect_water(self, generated_ids, high_entropy_positions): 
        codeword_len = len(generated_ids)
        if(self.prc_exists(codeword_len) == False):
            print(f"PRC of Codeword Length {codeword_len} does not exist.")
            return False
        bit_str = [self.hash_fn(generated_ids[i]) for i in range(codeword_len)]
        bit_str = np.fromiter(bit_str, dtype = np.uint8, count = codeword_len)
        prc = self.load_prc(codeword_len)
        return prc.is_codeword(bit_str, false_positive_rate = 0.0)

    
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
        

    