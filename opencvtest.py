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
import math

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

    def detect_color(self,image,contour):
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.fillPoly(mask, pts =[contour], color=(255)) # fill the polygon
        colorimage = cv2.bitwise_and(image, mask)
        colorindexed = colorimage[int(colorimage.shape[0]/2),int(colorimage.shape[0]/2)]
        COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
        colorindex = COLOR_VALUES.index(colorindexed)
        color_values = ['light yellow', 'yellow', 'light green', 'green', 'jade' ,'greenish blue', 'dark blue', 'blue', 'purple', 'dark purple']
        color = color_values[colorindex]
        return color
    
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
        return size    
    
    def detect_shape(self, images):
        # converting image into grayscale image
        shapes = []
        for image in images:
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
            color = self.detect_color(image,contour)
            size = self.detect_size(image,contour)
            shapes.append([shape,color,size])

        return shapes

    def forward(self, images):
        return self.detect_shape(images)

def inference(dataset, model):
    predclass, predsize, predcolor = [], [], []
    groundclass,groundsize, groundcolor  = [] , [], []
    predictions, groundtruths = [], []
    types = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
    colors = ['light yellow', 'yellow', 'light green', 'green', 'jade' ,'greenish blue', 'dark blue', 'blue', 'purple', 'dark purple']
    size = [0.4,0.5,0.6,0.7,0.8,0.9]
    # print("Inference of the model")
    for index, image_groundtruth in enumerate(zip(dataset.images, dataset.groundtruthshapes)):
        image, groundtruth = image_groundtruth
        out = model.forward(image)
        typelist = [types[g[0]] for g in groundtruth]
        colorlist = [colors[g[1]] for g in groundtruth]
        sizelist = [size[g[2]] for g in groundtruth]
        groundtruthclass = []
        for i in range(len(typelist)):
            groundtruthclass.append([typelist[i],colorlist[i],sizelist[i]])
        predictions.append(out)
        groundtruths.append(groundtruthclass)
        groundclass.append(typelist)
        groundsize.append(sizelist)
        groundcolor.append(colorlist)
        predclass.append([o[0] for o in out])
        predsize.append([o[2] for o in out])
        predcolor.append([o[1] for o in out])
    np.savez('OpenCV_test_complete', predictions=np.array(predictions), targets=np.array(groundtruths))
    return groundclass,groundsize, groundcolor, predclass, predsize, predcolor


def main():
    print("Loading dataset")
    dataset = Raven()       
    dataset.forward()
    print("Loading Model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = openCV_shape()
    types = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
    colors = ['light yellow', 'yellow', 'light green', 'green', 'jade' ,'greenish blue', 'dark blue', 'blue', 'purple', 'dark purple']
    size = [0.4,0.5,0.6,0.7,0.8,0.9]
    groundclass,groundsize, groundcolor, predclass, predsize, predcolor = inference(dataset, model)
    print(classification_report(groundclass, predclass, labels=types))
    print(classification_report(groundsize, predsize, labels=size))
    print(classification_report(groundcolor, predcolor, labels=colors))


if __name__ == '__main__':
    main()

  
