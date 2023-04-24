import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
print(torch.__version__)

url = 'https://media.newyorker.com/cartoons/63dc6847be24a6a76d90eb99/master/w_1160,c_limit/230213_a26611_838.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16)
model.to(device)


inputs = processor(image, return_tensors="pt").to("cpu", torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)



