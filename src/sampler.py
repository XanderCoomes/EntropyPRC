class Sampler: 
    def __init__(self, model, tokenizer, temperature, no_repeat_ngram_size, repetition_penalty, top_p): 
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
    
    def logits(self, tokens): 
        pass

    def apply_no_repeat_ngram(self, tokens, logits):
        pass

    def apply_repetition_penalty(self, tokens, logits):
        pass

    def apply_temperature(self, logits):
        pass

    def probs(self, logits):
        pass

    def apply_top_p(self, probs):
        pass

    def sample(self, probs): 
        pass






        


    

    


    
