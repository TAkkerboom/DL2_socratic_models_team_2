import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision import transforms
from .const import SHAPES, ANGLES, SIZES, COLORS, NAME_TO_COLOR
import cv2

IMAGE_SIZE = 160

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
        
        for i, image_array in enumerate(images):
            image = Image.fromarray(image_array.astype(np.uint8), mode='L')
            image = image.convert('RGB')
            
            width, height = image.size
            
            draw = ImageDraw.Draw(image)
            
            for y in range(height):
                for x in range(width):
                    if image_array[y, x] != 255:
                        image.putpixel((x, y), item.colors[i])

            # Draw the x-axis (red line)
            draw.line([(0, height // 2), (width - 1, height // 2)], fill=(0, 0, 0), width=1)

            # Draw the y-axis (red line)
            draw.line([(width // 2, 0), (width // 2, height - 1)], fill=(0, 0, 0), width=1)

            puzzle.append(image)
            
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

            
class Item:
    def __init__(self, npz_path, xml_path, id):
        self.xml_path = xml_path
        self.npz_path = npz_path
        self.puzzle_id = id
        
        npz_data = np.load(self.npz_path)
        self.images = npz_data['image']
        self.target_id = npz_data['target']
        self.symbolic = {}
        self.colors = []
        self.puzzle_image = None
        
    def get_ground_truth(self):
        # Parse the xml file
        tree = ET.parse(self.xml_path)
        # Access the root of the xml tree
        root = tree.getroot()
        
        for i in range(16):
            data = root[0][i][0][0][0][0].attrib
            symbolic = {}
            
            symbolic['Type'] = SHAPES[data['Type']]
            symbolic['Color'] = COLORS[data['Color']]
            symbolic['Angle'] = ANGLES[data['Angle']]            
            symbolic['Size'] = SIZES[data['Size']]
            
            self.colors.append(NAME_TO_COLOR[symbolic['Color']])
            self.symbolic[i] = symbolic
                
    def generate_matrix(self, array_list):
        # row-major array_list
        array_list = array_list[0:8, :, :]
        img_grid = np.zeros((IMAGE_SIZE * 3, IMAGE_SIZE * 3), np.uint8)
        
        for idx in range(len(array_list)):
            i, j = divmod(idx, 3)
            img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = array_list[idx]
            
        # draw grid
        for x in [0.33, 0.67]:
            img_grid[int(x * IMAGE_SIZE * 3) - 1:int(x * IMAGE_SIZE * 3) + 1, :] = 0
        for y in [0.33, 0.67]:
            img_grid[:, int(y * IMAGE_SIZE * 3) - 1:int(y * IMAGE_SIZE * 3) + 1] = 0
            
        return img_grid

    def generate_answers(self, array_list):
        array_list = array_list[8:16, :, :]
        img_grid = np.zeros((IMAGE_SIZE * 2, IMAGE_SIZE * 4), np.uint8)
        
        for idx in range(len(array_list)):
            i, j = divmod(idx, 4)
            img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = array_list[idx]
            
        # draw grid
        for x in [0.5]:
            img_grid[int(x * IMAGE_SIZE * 2) - 1:int(x * IMAGE_SIZE * 2) + 1, :] = 0
        for y in [0.25, 0.5, 0.75]:
            img_grid[:, int(y * IMAGE_SIZE * 4) - 1:int(y * IMAGE_SIZE * 4) + 1] = 0
            
        return img_grid

    def generate_matrix_answer(self, array_list):
        # row-major array_list
        assert len(array_list) <= 18
        img_grid = np.zeros((IMAGE_SIZE * 6, IMAGE_SIZE * 3), np.uint8)
        
        for idx in range(len(array_list)):
            i, j = divmod(idx, 3)
            img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = array_list[idx]
            
        # draw grid
        for x in [0.33, 0.67, 1.00, 1.33, 1.67]:
            img_grid[int(x * IMAGE_SIZE * 3), :] = 0
        for y in [0.33, 0.67]:
            img_grid[:, int(y * IMAGE_SIZE * 3)] = 0
            
        return img_grid
    
    def merge_matrix_answer(self, array_list):
        matrix_image = self.generate_matrix(array_list)
        answer_image = self.generate_answers(array_list)
        
        img_grid = np.ones((IMAGE_SIZE * 5 + 20, IMAGE_SIZE * 4), np.uint8) * 255
        img_grid[:IMAGE_SIZE * 3, int(0.5 * IMAGE_SIZE):int(3.5 * IMAGE_SIZE)] = matrix_image
        img_grid[-(IMAGE_SIZE * 2):, :] = answer_image
        
        return img_grid

    def process(self):
        array = self.images
        self.merged = self.merge_matrix_answer(array)
        self.puzzle_image = self.generate_matrix(array[:8])
        
        self.get_ground_truth()
        
    def plot(self):
        name = self.npz_path.split('/')[-1]
        
        plt.title(f"Puzzle: {name}")
        plt.imshow(self.merged)
        plt.axis('off')
        plt.show()
