class WaterLLM: 
    def __init__(self, sampler, prc): 
        self.sampler = sampler
        self.prc = prc

    def dry_response(self, prompt):
        generated_ids = self.sampler.generate(prompt)
        return generated_ids
    
    def water_response(self, prompt):
        codeword = self.prc.encode(noise_rate = 0.0)

        
       
    