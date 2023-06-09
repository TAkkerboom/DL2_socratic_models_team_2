from transformers import BlipProcessor, BlipForQuestionAnswering, AutoTokenizer, CLIPProcessor, CLIPModel
import torch
import cv2
import math
import numpy as np
from .const import COLORS


class VLM:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device="cpu"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.model.to(device)

    def forward(self, image, prompt):
        with torch.no_grad():
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=50)
            out = self.processor.decode(out[0], skip_special_tokens=True)

        return out


class CLIP:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(device)

    def forward(self, image, prompt):
        with torch.no_grad():
            inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            label = prompt[torch.argmax(probs)]

        return label


class OpenCV:
    def __init__(self):
        pass

    def detect_color(self, image):
        COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
        colorindexed = image[int(image.shape[0]/2),int(image.shape[1]/2)]
        colorvalue = COLORS[str(COLOR_VALUES.index(colorindexed))]
        return colorvalue

    def detect_size(self,image,contour):
        area = cv2.contourArea(contour)
        imagesize = image.shape[0]*image.shape[1]
        size = int(area)/int(imagesize) + 0.3
        if size >0.9:
            size = 0.9
        elif size<0.4:
            size = 0.4
        else:
            size = math.ceil(size*10)/10
        return str(size)

    def detect_shape(self, image):
        # converting image into grayscale image
        shapes = []
        # setting threshold of gray image
        _, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # using a findContours() function
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # list for storing names of shapes
        contour = contours[-1]

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "square"
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 6:
            shape = "hexagon"
        else:
            shape = "circle"
        color = self.detect_color(image)
        size = self.detect_size(image, contour)
        angle = 0
        shapes = [angle, color, size, shape]

        return shapes

    def forward(self, image):
        return self.detect_shape(image)


class LM:
    def __init__(self, model, model_class, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model, device_map="auto")
        self.model = model_class.from_pretrained(model) # for t5
        self.model.to(device)

    def forward(self, batch):
        # inputs = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True).items()}
        outputs = self.model.generate(**inputs, max_length=512).to(self.device)  # max_length=512 for t5

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
