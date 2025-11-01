import torch
import torch.nn.functional as F

class Sampler: 
    def __init__(self, model, tokenizer, temperature, repetition_penalty, top_p): 
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p

    def calc_probs(self, generated_token_ids, past_key_vals):
        logits, past_key_vals = self.calc_logits(generated_token_ids, past_key_vals)
        logits = self.apply_repetition_penalty(generated_token_ids, logits)
        logits = self.apply_temperature(logits)
        probs = self.softmax(logits)
        probs = self.apply_top_p(probs)
        return probs, past_key_vals
        
    def calc_logits(self, generated_token_ids, past_key_vals): 
        self.model.eval()
        generated_token_ids = generated_token_ids.to(self.device)
        with torch.no_grad():
            if(past_key_vals == None):
                out = self.model(input_ids = generated_token_ids, use_cache = True)
            else:
                prev_token_id = generated_token_ids[:, -1:]
                out = self.model(input_ids = prev_token_id, past_key_values = past_key_vals, use_cache = True)
        logits = out.logits[:,-1,:]
        logits = logits.squeeze(0)
        return logits, out.past_key_values
    
    def apply_repetition_penalty(self, generated_ids, logits):
        for token_id in set(generated_ids[0].tolist()):
            logits[token_id] /= self.repetition_penalty
        return logits

    def apply_temperature(self, logits):
        logits = logits / self.temperature
        return logits
    
    def softmax(self, logits):
        probs = F.softmax(logits, dim = -1)
        return probs
    
    def apply_top_p(self, probs):
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)

        cutoff = torch.roll(cumsum > self.top_p, shifts=1, dims=-1)
        cutoff[..., 0] = False 

        mask = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, sorted_idx, cutoff)

        filtered = probs.masked_fill(mask, 0.0)
        return F.normalize(filtered, p = 1, dim = -1)

    def sample(self, probs): 
        next_token = torch.multinomial(probs.float(), num_samples = 1) 
        token_id = next_token.item()
        return token_id
    
    def text_to_ids(self, text): 
        input_batch = self.tokenizer(text, return_tensors = "pt", add_special_tokens = False)
        generated_ids = input_batch["input_ids"]
        return generated_ids
    
    
    





        


    

    


    
