from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

class VLM:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device="cpu"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        
    def forward(self, image, prompt):
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_length=50)
        out = self.processor.decode(out[0], skip_special_tokens=True)
        
        return out  