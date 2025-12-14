from prcwatermark.sampler import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SamplerTester: 
    def __init__(self, sampler): 
        self.sampler = sampler
    def test_sampling(self):
        generated_ids = self.sampler.txt_to_ids("You are a helpful educational assistant, please write an essay on birds.")
        past_key_values = None
        for _ in range(100):
            probs, past_key_values = self.sampler.calc_probs(generated_ids, past_key_values)
            next_token_id = self.sampler.sample(probs)
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=-1)
            next_token = self.sampler.tokenizer.decode([next_token_id], skip_special_tokens = True)
            print(next_token, end = '', flush = True)
        print()

        text = self.sampler.ids_to_txt(generated_ids[0])
        ids = self.sampler.txt_to_ids(text)

        assert(ids == generated_ids).all(), "Text to IDs conversion failed."
    
if __name__ == "__main__":
    temperature = 1.5
    repetition_penalty = 1.0
    top_p = 0.8
    top_k = 10

    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = "auto", device_map = "auto")

    sampler = Sampler(model, tokenizer, temperature, repetition_penalty, top_p, top_k)
    tester = SamplerTester(sampler)
    tester.test_sampling()

