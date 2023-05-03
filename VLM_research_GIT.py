from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
import numpy as np
from os import listdir
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET
from plot_raven_mod import plot_raven
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
        endimage = len(self.npz_data)
        for npz, xml in zip(self.npz_data[:endimage], self.xml_data[:endimage]):
            data_point = Dataloader('{}/{}'.format(self.path, npz), '{}/{}'.format(self.path, xml))
            data_point.process()
            self.images.append(data_point.ravensmall)
            self.groundtruthshapes.append(data_point.groundtruthshapes)
            i += 1
            print("Loading image :{}".format(i))
        pass


class VLM:   
    def __init__(self, device):
        # print("Loading VLM")
        self.device = device
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")
   
    def forward_caption(self, image, text):
        # print("Create caption")
        encoding = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        self.model.to(self.device)
        input_ids = self.processor(text=text, add_special_tokens=False).input_ids
        input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(pixel_values=encoding, input_ids=input_ids, max_length=50)
        return self.processor.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
    
    def forward(self, image):
        text = "How many vertices does the geometric shape have?"
        textbatch = [text for i in range(len(image))]
        answer = []
        for i in range(len(textbatch)):
            answer.append(self.forward_caption(image[i], textbatch[i])[0])
        return answer

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
        outnew = out.copy()
        for iout, predout in enumerate(out):
            if predout.isnumeric():
                if int(predout) == 0:
                    outnew[iout] = "circle"
                elif int(predout[0]) == 3:
                    outnew[iout] = "triangle"
                elif int(predout) == 4:
                    outnew[iout] = "square"
                elif int(predout) == 5:
                    outnew[iout] = "pentagon"
                elif int(predout) == 6:
                    outnew[iout] = "hexagon"
                else:
                    outnew[iout] = "none"
            else:
                outnew[iout] = "none"
        groundtruthclass = [types[g[0]] for g in groundtruth]
        count += sum(a == b for a, b in zip(outnew, groundtruthclass))
        print(f"The count is equal to:{count} for image {index}")
        for output,gclass in zip(outnew,groundtruthclass):
            predictions.append(output)
            groundtruths.append(gclass)
    np.savez('GIT_performance_ask_vertices', predictions=np.array(predictions), targets=np.array(groundtruths))

    return groundtruths, predictions


def main():
    print("Loading dataset")
    dataset = Raven()       
    dataset.forward()
    print("Loading Model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VLM(device)
    types = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
    groundtruths, pred = inference(dataset, model)
    print(classification_report(groundtruths, pred, labels=types))


if __name__ == '__main__':
    main()