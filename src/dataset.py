import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.const import SHAPES, ANGLES, SIZES, COLORS


class Raven:
    def __init__(self, root='./dataset/RAVEN-10000/', split='test', fig_type='*'):
        self.npz_data = [f for f in glob.glob(os.path.join(root, fig_type, '*.npz')) if split in f]
        self.xml_data = [f for f in glob.glob(os.path.join(root, fig_type, '*.xml')) if split in f]
        self.split = split
        self.fig_type = fig_type
        self.items = []
        
    def len(self):
        return len(self.items)
    
    def get_puzzle(self, id):
        item = self.items[id]
        images = item.images[0:16, :, :]
        puzzle = []
        
        for image in images:
            img = Image.fromarray(image).convert("P")
            puzzle.append(img)
            
        return puzzle

    def plot_puzzle(self, id):
        item = self.items[id]
        item.plot()

    def get_answers(self, id):
        item = self.items[id]
        
        return item.symbolic
    
    def load_data(self):
        for i, (npz, xml) in enumerate(zip(self.npz_data, self.xml_data)):
            item = Item(npz, xml, i)
            item.process()
            
            self.items.append(item)
            print(f"Loading item: {i}")
            break
            
class Item:
    def __init__(self, npz_path, xml_path, id):
        self.xml_path = xml_path
        self.npz_path = npz_path
        self.puzzle_id = id
        
        npz_data = np.load(self.npz_path)
        self.images = npz_data['image']
        self.target_id = npz_data['target']
        self.symbolic = {}
        
    def get_image(self):        
        # Create empty array to store the images
        grid = np.zeros((480, 480))
        
        for j in range(8):
            # Extract each image from the dataset
            img = self.images[j]
            # Calculate the starting index of each image
            start_i = (j // 3) * 160
            start_j = (j % 3) * 160
            # Get the ending index of each image
            end_i = start_i + 160
            end_j = start_j + 160
            # Place the image in the empty array
            grid[start_i:end_i, start_j:end_j] = img
            
        self.grid = grid

    def get_ground_truth(self):
        # Parse the xml file
        tree = ET.parse(self.xml_path)
        # Access the root of the xml tree
        root = tree.getroot()
        
        for i in range(16):
            data = root[0][i][0][0][0][0].attrib
            symbolic = {}
            
            symbolic['Angle'] = ANGLES[data['Angle']]
            symbolic['Color'] = COLORS[data['Color']]
            symbolic['Size'] = SIZES[data['Size']]
            symbolic['Type'] = SHAPES[data['Type']]
            
            self.symbolic[i] = symbolic
        
    def plot(self):
        name = self.npz_path.split('/')[-1]
        
        plt.title(f"Puzzle: {name}")
        plt.imshow(self.grid)
        plt.axis('off')
        plt.show()

        fig, axes = plt.subplots(1, 8, figsize=(20, 15))
        plt.title(f"Answers: {name}")
        for j in range(8):
            axes[j].imshow(self.images[-8+j, :, :])
            axes[j].set_axis_off()
            
        plt.axis('off')
        plt.show()

        self.grid[320:480, 320:480] = self.images[8+self.target_id]
        plt.title(f'Solution:{name}')
        plt.imshow(self.grid)
        plt.axis('off')
        plt.show()
    
    def process(self):
        self.get_image()
        self.get_ground_truth()