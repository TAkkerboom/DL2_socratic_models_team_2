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
            # angle = int(entity.get('Angle'))
            color = int(entity.get('Color'))
            size = int(entity.get('Size'))
            #print the variables
            groundtruthshapes.append([type,color,size])
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
        endimage = 2
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

    def detect_color(self,image,contour):
        maskImage = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(maskImage, contour, -1, (255, 255, 255), -1)
        colorimage = cv2.bitwise_not(image, maskImage)
        max_color = np.amax(colorimage)
        COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
        colorindex = COLOR_VALUES.index(max_color)
        color_values = ['light yellow', 'yellow', 'light green', 'green', 'jade' ,'greenish blue', 'dark blue', 'blue', 'purple', 'dark purple']
        color = color_values[colorindex]
        return color
    
    def detect_size(self,image,contour):
        area = cv2.contourArea(contour)
        imagesize = image.shape[0]*image.shape[1]
        size = int(area)/int(imagesize)
        size = round(size, 1)

        return size    
    
    def detect_shape(self, images):
        # converting image into grayscale image
        shapes = []
        for image in images:
            cvimage = np.asarray(image)
            gray = cv2.cvtColor(cvimage, cv2.COLOR_BGR2GRAY)
            
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
                shape = "triangle"
            elif len(approx) == 4:
                shapes.append("square")
                shape = "square"
            elif len(approx) == 5:
                shapes.append("pentagon")
                shape = "pentagon"
            elif len(approx) == 6:
                shapes.append("hexagon")
                shape = "hexagon"
            else:
                shapes.append("circle")
                shape = "circle"
            color = self.detect_color(cvimage,contour)
            size = self.detect_shape(cvimage,contour)
            shapes.append([shape,color,size])

        return shapes

    def forward(self, images):
        return self.detect_shape(images)

def inference(dataset, model):
    count = 0
    predictions = []
    groundtruths = []
    types = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
    colors = ['light yellow', 'yellow', 'light green', 'green', 'jade' ,'greenish blue', 'dark blue', 'blue', 'purple', 'dark purple']
    size = [0.4,0.5,0.6,0.7,0.8,0.9]
    # print("Inference of the model")
    count = 0 
    for index, image_groundtruth in enumerate(zip(dataset.images, dataset.groundtruthshapes)):
        image, groundtruth = image_groundtruth
        convertimage = [Image.fromarray(single_image).convert("RGB") for single_image in image]
        out = model.forward(convertimage)
        typelist = [types[g] for g[0] in groundtruth]
        colorlist = [colors[g] for g[1] in groundtruth]
        sizelist = [sizes[g] for g[2] in groundtruth]
        groundtruthclass = []
        for i in range(len(typelist)):
            groundtruthclass.append([typelist[i],colorlist[i],sizelist[i]])
        count += sum(a == b for a, b in zip(out, groundtruthclass))
        predictions.append(out)
        groundtruths.append(groundtruthclass)
    np.savez('OpenCV_test_complete', predictions=np.array(predictions), targets=np.array(groundtruths))

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
  
