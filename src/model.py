from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import CLIPProcessor, CLIPModel
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
    
class CLIP(VLM):
    def forward(self, image, prompt):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        return probs