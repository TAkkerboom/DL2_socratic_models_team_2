from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import numpy as np
from os import listdir
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET
from plot_raven import plot_raven
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
        endimage = 100
        for npz, xml in zip(self.npz_data[:endimage], self.xml_data[:endimage]):
            data_point = Dataloader('{}/{}'.format(self.path, npz), '{}/{}'.format(self.path, xml))
            data_point.process()
            self.images.append(data_point.ravensmall)
            self.groundtruthshapes.append(data_point.groundtruthshapes)
            i += 1
            print("Loading image :{}".format(i))
        pass


class VLM:   
    def __init__(self):
        # print("Loading VLM")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
   
    def forward_caption(self, image, text):
        # print("Create caption")
        encoding = self.processor(image, text, return_tensors="pt")
        outputs = self.model(**encoding)
        logits = outputs.logits
        idxs = [logit.argmax(-1).item() for logit in logits]
        return [self.model.config.id2label[idx] for idx in idxs]

    def forward(self, image):
        text = "What is the shape in the object?"
        textbatch = [text for i in range(len(image))]
        return self.forward_caption(image, textbatch)

def inference(dataset, model):
    count = 0
    predictions = []
    groundtruths = []
    types = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
    # print("Inference of the model")
    count = 0 
    for index, image_groundtruth in enumerate(zip(dataset.images, dataset.groundtruthshapes)):
        image, groundtruth = image_groundtruth
        convertimage = [Image.fromarray(single_image).convert("RGB") for single_image in image]
        out = model.forward(convertimage)
        groundtruthclass = [types[g[0]] for g in groundtruth]
        count += sum(a == b for a, b in zip(out, groundtruthclass))
        print(f"The count is equal to:{count} for image {index}")
        for output,gclass in zip(out,groundtruthclass):
            predictions.append(output)
            groundtruths.append(gclass)
    np.savez('VilT_performance', predictions=np.array(predictions), targets=np.array(groundtruths))

    return groundtruths, predictions


def main():
    print("Loading dataset")
    dataset = Raven()       
    dataset.forward()
    print("Loading Model")
    model = VLM()
    types = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
    groundtruths, pred = inference(dataset, model)
    print(classification_report(groundtruths, pred, labels=types))


if __name__ == '__main__':
    main()