import pickle
from pathlib import Path
import torch
from prcwatermark.prc import PRC

class WaterLLM: 
    def __init__(self, sampler, hash_fn, sparsity_fn, key_folder): 
        self.sampler = sampler
        self.hash_fn = hash_fn
        self.sparsity_fn = sparsity_fn
        self.key_folder = key_folder

    def gen_response(self, prompt, num_tokens, is_water):
        generated_ids = self.sampler.text_to_ids(prompt)
        past_key_values = None
        for _ in range(num_tokens):
            probs, past_key_values = self.sampler.calc_probs(generated_ids, past_key_values)
            next_token_id = self.sampler.sample(probs)
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=-1)
            next_token = self.sampler.tokenizer.decode([next_token_id])
            print(next_token, end = '', flush = True)
        
        response = self.sampler.ids_to_text(generated_ids)
        return response

    def detect_water (self, response): 
        pass

    def prc_file_name(self, codeword_len):
        return str(codeword_len) + "bit_prc.pkl"
    
    def save_prc(self, prc):
        folder = Path(self.key_folder)
        folder.mkdir(parents = True, exist_ok = True)
        with open(folder / self.prc_file_name(prc.codeword_len), "wb") as f:
            pickle.dump(prc, f, protocol = pickle.HIGHEST_PROTOCOL)

    def load_prc(self, codeword_len):
        with open(Path(self.key_folder) / self.prc_file_name(codeword_len), "rb") as f:
            return pickle.load(f)

    