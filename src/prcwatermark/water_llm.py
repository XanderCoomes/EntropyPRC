import torch

class WaterLLM: 
    def __init__(self, sampler): 
        self.sampler = sampler
    
    def gen_response(self, prompt, num_tokens, is_water):
        generated_ids = self.sampler.text_to_ids(prompt)
        past_key_values = None
        for _ in range(num_tokens):
            probs, past_key_values = self.sampler.next_token_probs(generated_ids, past_key_values)
            next_token_id = self.sampler.sample(probs)
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=-1)
            next_token = self.sampler.tokenizer.decode([next_token_id])
            print(next_token, end = '', flush = True)

        

    def detect_water (self, prompt): 
        pass

    