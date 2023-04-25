from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
from os import listdir
import os
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET


class Preprocess:

    def __init__(self, npz_path, xml_path):
        self.xml_path = xml_path
        self.npz_path = npz_path

    def get_target(self):
        data = np.load(self.npz_path)
        self.target = data['target'].item()

    def get_attributes(self):
        #parse the xml file
        tree = ET.parse(self.xml_path)

        #access the root of the xml tree
        root = tree.getroot()

        attributes = []
        #iterate through the panels
        for panel in root[0].findall('Panel'):
            #find the component 
            component = panel.find('Struct/Component')
            #find the entity
            entity = component.find('Layout/Entity')
            #access the variables
            angle = int(entity.get('Angle'))
            color = int(entity.get('Color'))
            size = int(entity.get('Size'))
            type = int(entity.get('Type'))
            #print the variables
            attributes.append([angle, color, size, type])
        self.attributes = attributes

    def process(self):
        self.get_target()
        self.get_attributes()


class Dataloader:

    def __init__(self, path):
        self.path = path
        self.attributes = []
        self.targets = []

    def load(self):
        prep = np.load(self.path)
        self.attributes = prep['attributes']
        self.targets = prep['targets']



class Raven:

    def __init__(self, dir='center_single'):
        self.path = 'RAVEN-10000/{}'.format(dir)
        self.filenames = listdir(self.path)
        self.npz_data = [file for file in self.filenames if file.endswith('.npz')]
        self.xml_data = [file for file in self.filenames if file.endswith('.xml')]
        self.attributes = []
        self.prompts = []
        self.targets = []
        self.descriptions = []

    def generate_single_description(self, angle, color, size, type):
        COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
        color_values = ['light yellow', 'yellow', 'light green', 'green', 'jade' ,'greenish blue', 'dark blue', 'blue', 'purple', 'dark purple']
        ANGLE_VALUES = [-135, -90, -45, 0, 45, 90, 135, 180]
        TYPE_VALUES = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
        template = 'a {} {} of size {} at a {} degrees angle'.format(color_values[color], 
                                                            TYPE_VALUES[type], 
                                                            size, 
                                                            str(ANGLE_VALUES[angle]))
        return template

    def get_descriptions(self, attributes):
        descriptions = []
        for angle, color, size, type in attributes:
            descriptions.append(self.generate_single_description(angle, color, size, type))
        return descriptions

    def generate_prompt(self, descriptions):
        prompt = '''You are given a logic puzzle from the RAVEN dataset. The first shape on the first row is {}, the second shape on the first row is {}, the third item on the first row is {}. The first shape on the second row is {}, the second shape on the second row is a {}, the third shape on the second row is {}. The first shape on the third row is {}, the second shape is {}. Based on this, what is the third shape on the third row? You can only choose between: {}, {}, {}, {}, {}, {}, {}, {}'''.format(*descriptions)
        return prompt
    
    def forward(self):
        if os.path.isfile('center_single_prep.npz'):
            loader = Dataloader('center_single_prep.npz')
            loader.load()
            self.attributes = loader.attributes
            self.targets = loader.targets
            for data_point in self.attributes:
                descriptions = self.get_descriptions(data_point)
                self.prompts.append(self.generate_prompt(descriptions))
                self.descriptions.append(descriptions)
        else:
            for npz, xml in zip(self.npz_data, self.xml_data):
                data_point = Preprocess('{}/{}'.format(self.path, npz), '{}/{}'.format(self.path, xml))
                data_point.process()
                self.attributes.append(data_point.attributes)
                descriptions = self.get_descriptions(data_point.attributes)
                self.targets.append(int(data_point.target))
                self.prompts.append(self.generate_prompt(descriptions))
                self.descriptions.append(descriptions)
                
            # Save the data on disk
            np.savez('center_single_prep', attributes=np.array(self.attributes), targets=np.array(self.targets))
        pass


class LM:
    
    def __init__(self, model):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
    def forward(self, input_str):
        inputs = self.tokenizer(input_str, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=50)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    

def inference(dataset, model):
    count = 0
    predictions = []
    for i, data in enumerate(zip(dataset.prompts, dataset.targets)):
        prompt, target = data
        out = model.forward(prompt)[0]
        pred = dataset.descriptions[i][8:].index(out)
        count += int(pred == target)
        predictions.append(pred)
    return count, predictions


def main():
    dataset = Raven()  
    print('loading data...')     
    dataset.forward()
    print('loading model...')
    model = LM("google/flan-t5-large")
    print('running ... (this may take some time)')
    count, pred = inference(dataset, model)
    print(classification_report(dataset.targets, pred))


if __name__ == '__main__':
    main()
