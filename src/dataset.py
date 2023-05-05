import os
import glob
import numpy as np
import gc
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class Raven:
    def __init__(self, root='./dataset/RAVEN-10000/', split='test', fig_type='*'):
        
        self.npz_data = [f for f in glob.glob(os.path.join(root, fig_type, '*.npz')) if split in f]
        self.xml_data = [f for f in glob.glob(os.path.join(root, fig_type, '*.xml')) if split in f]
        self.data = []
        
    def len(self):
        return len(self.npz_data)
    
    def load_data(self):
        for i, (npz, xml) in enumerate(zip(self.npz_data, self.xml_data)):
            item = Dataloader(npz, xml, i)
            item.process()
            
            self.data.append(item)
            print(f"Loading item: {i}")
            break
            

class Dataloader:
    def __init__(self, npz_path, xml_path, id):
        self.xml_path = xml_path
        self.npz_path = npz_path
        self.id = id
        
    def get_image(self):
        self.data = np.load(self.npz_path)
        
        # Create empty array to store the images
        grid = np.zeros((480, 480))
        images = []
        
        # Create a loop to go through each image
        for j in range(8):
            # Extract each image from the dataset
            img = self.data['image'][j]
            # Reshape each image to (160, 160)
            img = img.reshape(160, 160)
            # Calculate the starting index of each image
            start_i = (j // 3) * 160
            start_j = (j % 3) * 160
            # Get the ending index of each image
            end_i = start_i + 160
            end_j = start_j + 160
            # Place the image in the empty array
            grid[start_i:end_i, start_j:end_j] = img
            images.append(img)
            
        self.images, self.grid = images, grid

    def ground_truth_shape(self):
        # parse the xml file
        tree = ET.parse(self.xml_path)

        # access the root of the xml tree
        root = tree.getroot()

        shapes = []
        
        # iterate through the panels
        for panel in root[0].findall('Panel'):
            # find the component 
            component = panel.find('Struct/Component')
            # find the entity
            entity = component.find('Layout/Entity')
            # access the variables
            type = int(entity.get('Type'))
            # print the variables
            shapes.append([type])
            
        self.shapes = shapes[:8]
        
    def plot(self):
        # Plot the final image
        plt.title(f"Puzzle {self.id}")
        plt.imshow(self.grid)
        plt.axis('off')
        plt.show()

        fig, axes = plt.subplots(1, 8, figsize=(20, 15))
        plt.title(f"Answers {self.id}")
        for j in range(8):
            axes[j].imshow(self.data['image'][-8+j, :, :])
            axes[j].set_axis_off()
            
        plt.axis('off')
        plt.show()

        self.grid[320:480, 320:480] = self.data['image'][8+self.data['target']]
        plt.title(f'Solution {self.id}')
        plt.imshow(self.grid)
        plt.axis('off')
        plt.show()
    
    def process(self):
        self.get_image()
        self.ground_truth_shape()