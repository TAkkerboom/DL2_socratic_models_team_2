from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import numpy as np
from os import listdir
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET
from plot_raven_mod import plot_raven
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

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
            groundtruthshapes.append(type)
        self.groundtruthshapes = groundtruthshapes[:8]
    
    def process(self):
        self.get_image()
        self.ground_truth_shape()


class Raven:

    def __init__(self, dir='center_single'):
        self.path = 'RAVEN-10000/{}'.format(dir)
        self.filenames = sorted(listdir(self.path))
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


class openCV_shape:   
    def __init__(self):
        pass

    def detect_shape(self, images):
        # converting image into grayscale image
        shapes = []
        for image in images:
            gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
            
            # setting threshold of gray image
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # using a findContours() function
            contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                     
            # list for storing names of shapes
            contour = contours[-1]
                   
            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) == 3:
                shapes.append("triangle")
                print("triangle")
            elif len(approx) == 4:
                shapes.append("square")
                print("square")
            elif len(approx) == 5:
                shapes.append("pentagon")
                print("pentagon")
            elif len(approx) == 6:
                shapes.append("hexagon")
                print("hexagon")
            else:
                shapes.append("circle")
                print("circle")
        return shapes

    def forward(self, images):
        return self.detect_shape(images)

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
        groundtruthclass = [types[g] for g in groundtruth]
        count += sum(a == b for a, b in zip(out, groundtruthclass))
        print(f"The count is equal to:{count} for image {index}")
        for output,gclass in zip(out,groundtruthclass):
            predictions.append(output)
            groundtruths.append(gclass)
    np.savez('OpenCV_test', predictions=np.array(predictions), targets=np.array(groundtruths))

    return groundtruths, predictions


def main():
    print("Loading dataset")
    dataset = Raven()       
    dataset.forward()
    print("Loading Model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = openCV_shape()
    types = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
    groundtruths, pred = inference(dataset, model)
    print(classification_report(groundtruths, pred, labels=types))


if __name__ == '__main__':
    main()


  
# reading image
img = cv2.imread('shapes.png')
  
