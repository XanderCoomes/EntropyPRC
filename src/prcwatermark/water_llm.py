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
        
    def gen_response(self, prompt, num_tokens, is_water):
        codeword = self.sample_codeword(num_tokens)
        print(codeword)
        generated_ids = self.sampler.text_to_ids(prompt)
        prompt_tokens = generated_ids.squeeze(0).shape[0]
        past_key_values = None

        for i in range(num_tokens):
            probs, past_key_values = self.sampler.calc_probs(generated_ids, past_key_values)
            if(is_water): 
                probs = self.bias_probs(probs, codeword[i])
            next_token_id = self.sampler.sample(probs)
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=-1)
            next_token = self.sampler.tokenizer.decode([next_token_id])
            if(self.hash_fn(next_token_id) == codeword[i]):
                print(f"\033[42m{next_token}\033[0m", end = '', flush = True)
            else:
                print(f"\033[41m{next_token}\033[0m", end = '', flush = True)
        
        response = self.sampler.ids_to_text(generated_ids[:, prompt_tokens:])
        return response

    def detect_water (self, response): 
        generated_ids = self.sampler.text_to_ids(response)
        generated_ids = generated_ids.squeeze(0)
        codeword_len = len(generated_ids)
        bit_str = [self.hash_fn(generated_ids[i]) for i in range(codeword_len)]
        bit_str = np.fromiter(bit_str, dtype = np.uint8, count = codeword_len)
        print(bit_str)
        prc = self.load_prc(codeword_len)
        return prc.decode(bit_str, false_positive_rate = 0)

    
    def prc_file_name(self, codeword_len):
        return str(codeword_len) + "bit_prc.pkl"
    
    def save_prc(self, prc):
        folder = Path(self.key_folder)
        folder.mkdir(parents = True, exist_ok = True)
        with open(folder / self.prc_file_name(prc.codeword_len), "wb") as f:
            pickle.dump(prc, f, protocol = pickle.HIGHEST_PROTOCOL)

  
    def load_prc(self, codeword_len): 
        file_name = self.prc_file_name(codeword_len)
        file_path = Path(self.key_folder) / file_name
        if(file_path.exists()):
            with open(Path(file_path), "rb") as f:
                prc =  pickle.load(f)
        else:
            prc = PRC(codeword_len, self.sparsity_fn)
            self.save_prc(prc)
        return prc
        

    