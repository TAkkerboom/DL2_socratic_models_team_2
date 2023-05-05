from transformers import BlipProcessor, BlipForQuestionAnswering

class VLM:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device="cpu"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
    
    def forward(self, image, prompt):
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        out = self.processor.decode(out[0], skip_special_tokens=True)
        
        return out