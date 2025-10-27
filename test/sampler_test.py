from prcwatermark.sampler import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SamplerTester: 
    def __init__(self, sampler): 
        self.sampler = sampler
    def test_sampling(self):
        generated_ids = self.sampler.text_to_ids("You are a helpful educationsl assistant, please write an essay on birds.")
        past_key_values = None
        for _ in range(100):
            logits, past_key_values = self.sampler.calc_logits(generated_ids, past_key_values)
            logits = self.sampler.apply_repetition_penalty(generated_ids, logits)
            logits = self.sampler.apply_temperature(logits)
            probs = self.sampler.softmax(logits)
            probs = self.sampler.apply_top_p(probs)
            next_token_id = self.sampler.sample(probs)
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=-1)
            next_token = self.sampler.tokenizer.decode([next_token_id])
            print(next_token, end = '', flush = True)
        print()

        text = self.sampler.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        ids = self.sampler.text_to_ids(text)
        assert(ids == generated_ids).all(), "Text to IDs conversion failed."
    
if __name__ == "__main__":
    temperature = 0.7
    repetition_penalty = 1.0
    top_p = 0.8

    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = "auto", device_map = "auto")

    sampler = Sampler(model, tokenizer, temperature, repetition_penalty, top_p)
    tester = SamplerTester(sampler)
    tester.test_sampling()

