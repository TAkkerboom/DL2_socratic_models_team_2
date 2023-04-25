from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import numpy as np
from os import listdir
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET
from plot_raven import plot_raven
import torch.nn as nn
from torchvision import transforms
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class Dataloader:

    def __init__(self, npz_path, xml_path):
        self.xml_path = xml_path
        self.npz_path = npz_path
        

    def get_image(self):
        self.ravensmall, self.raven = plot_raven(np.load(self.npz_path))

    def ground_truth_shape(self):
        #parse the xml file
        tree = ET.parse(self.xml_path)

        #access the root of the xml tree
        root = tree.getroot()

        groundtruthshapes = []
        #iterate through the panels
        for panel in root[0].findall('Panel'):
            #find the component 
            component = panel.find('Struct/Component')
            #find the entity
            entity = component.find('Layout/Entity')
            #access the variables
            type = int(entity.get('Type'))
            #print the variables
            groundtruthshapes.append([type])
        self.groundtruthshapes = groundtruthshapes[:8]
    
    def process(self):
        self.get_image()
        self.ground_truth_shape()


class Raven:

    def __init__(self, dir='center_single'):
        self.path = 'RAVEN-10000/{}'.format(dir)
        self.filenames = listdir(self.path)
        self.npz_data = [file for file in self.filenames if file.endswith('.npz')]
        self.xml_data = [file for file in self.filenames if file.endswith('.xml')]
        self.images = []
        self.groundtruthshapes = []
    
    def forward(self):
        i = 0
        for npz, xml in zip(self.npz_data, self.xml_data):
            data_point = Dataloader('{}/{}'.format(self.path, npz), '{}/{}'.format(self.path, xml))
            data_point.process()
            self.images.append(data_point.ravensmall)
            self.groundtruthshapes.append(data_point.groundtruthshapes)
            i += 1
            print("Loading image :{}".format(i))
            if i == 100:
                break
        pass


class VLM:   
    def __init__(self):
        print("Loading VLM")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        print("LM")
        self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
        self.modelT5 = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
    
    def run_model(self, input_string, max_new_tokens=50):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
            res = self.modelT5.generate(input_ids.to(self.model.device), max_new_tokens=max_new_tokens)
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)
        
    def forward_caption(self, image, text):
        print("Create caption")
        encoding = self.processor(image, text, return_tensors="pt")
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()

        return self.model.config.id2label[idx]
    
    def forward_multiple_choice(self, text, choices, caption):
        print("Create multiple choice")
        prompt = f"please describe this image according to the given question: {text}"
      
        if len(choices) > 0:
            choice_text = ""
            for i in range(len(choices)):
                choice_text += f"({i+1}) {choices[i]} "

        return self.run_model(f"{prompt}. The caption by the VLM is equal to: {caption}. What object is the closest to it according to the following choices: {choice_text}?")

    def forward(self, image):
        print("Final VLM prediction")
        text = "What is the shape in the object?"
        choices = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
        caption = self.forward_caption(image, text)
        return self.forward_multiple_choice(text, choices, str(caption))

def inference(dataset, model):
    count = 0
    predictions = []
    amount_of_crops = 8
    types = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
    print("Inference of the model")
    for i, image in enumerate(zip(dataset.images, dataset.groundtruthshapes)):
        image, groundtruth = image
        for j in range(amount_of_crops):
            out = model.forward(Image.fromarray(image[j]).convert("RGB"))[0]
            pred = types.index(out)
            count += int(pred == groundtruth[j][0])
            print(count)
            predictions.append(pred)
    return count, predictions


def main():
    print("Loading dataset")
    dataset = Raven()       
    dataset.forward()
    print("Loading Model")
    model = VLM()
    count, pred = inference(dataset, model)
    print(classification_report(dataset.groundtruthshapes, pred))


if __name__ == '__main__':
    main()