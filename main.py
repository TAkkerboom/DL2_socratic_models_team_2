from src.dataset import Raven
from src.model import VLM, LM, OpenCV, CLIP
from src.const import SHAPES, ANGLES, SIZES, COLORS
from transformers import AutoModelForSeq2SeqLM

def generate_single_description(angle, color, size, type):
    template = 'a {} {} of size {} at a {} degrees angle'.format(color, type, size, angle)
    return template


class SM:
    def __init__(self):
        self.PATH = "C:/Users/maart/Downloads/RAVEN-10000-release/RAVEN-10000"
        fig_types = ['center_single']

        self.test_set = Raven(self.PATH, 'test', fig_types[0])
        self.test_set.load_data()

        self.VLM = VLM()
        self.LM = LM("google/flan-t5-large", 'cpu', AutoModelForSeq2SeqLM)
        self.prompt = '''You are given a logic puzzle from the RAVEN dataset. The first shape on the first row is {}, the second shape on the first row is {}, the third item on the first row is {}. The first shape on the second row is {}, the second shape on the second row is a {}, the third shape on the second row is {}. The first shape on the third row is {}, the second shape is {}. Based on this, what is the third shape on the third row? You can only choose between: {}, {}, {}, {}, {}, {}, {}, {}'''


    def VLM_pred_attributes(self):
        answers = []

        for i in range(self.test_set.len()):
            puzzle = self.test_set.get_puzzle(i)
            puzzle_answers = []
            for j, c_image in enumerate(puzzle):
                angle = self.VLM.forward(c_image, 'At what angle is the figure located? The options are: -135, 90, -45, 0, 45, 90, 135, 180')
                color = self.VLM.forward(c_image, 'What is the color value of the figure?')
                size = self.VLM.forward(c_image, 'What is the size of the figure? The options are: 0.4, 0.5, 0.6, 0.7, 0.8, 0.9')
                type = self.VLM.forward(c_image, 'What is the shape of the figure? The options are: triangle, square, pentagon, hexagon, circle')
                puzzle_answers.append([angle, color, size, type])
                
            answers.append(puzzle_answers)

        return answers
    
    def OpenCV_pred_attributes(self):
        answers = []
        opencvmodel = OpenCV()
        for i in range(self.test_set.len()):
            puzzle = self.test_set.get_puzzle(i)
            puzzle_answers = []
            for j, c_image in enumerate(puzzle):
                puzzle_answers.append(opencvmodel.detect_shape(c_image))
            answers.append(puzzle_answers)
        return answers
            
    def generate_prompts(self, attributes):
        return [self.prompt.format(*[generate_single_description(*shape) for shape in puzzle]) for puzzle in attributes]

    def get_descriptions(self, attributes):
        descriptions = []
        for puzzle in attributes:
            shape_desc = []
            for shape in puzzle[-8:]:
                for angle, color, size, type in shape:
                    shape_desc.append(generate_single_description(angle, color, size, type))
            descriptions.append(shape_desc)
        return descriptions

    def inference(self, prompts, descriptions, batch_size, samples):
        predictions = []
        j = 0
        prompts = [prompts[x:x+batch_size] for x in range(0, len(prompts), batch_size)] 
        descriptions = [prompts[x:x+batch_size] for x in range(0, len(descriptions), batch_size)] 
        N  = int(samples / batch_size)
        for description_batch, prompt_batch in zip(descriptions[:N], prompts[:N]):
            out = self.LM.forward(prompt_batch)
            for output, description in zip(out, description_batch):
                if output in description:
                    pred = description.index(output)
                else:
                    pred = 8
                predictions.append(pred)
            j += 1
            if j % 1 == 0:
                print('currently at ', j * 10)
            
        return predictions

    def forward(self):
        attributes = self.VLM_pred_attributes()
        # attributes = self.OpenCV_pred_attributes()
        prompts = self.generate_prompts(attributes)
        descriptions = self.get_descriptions(attributes)
        self.inference(prompts, descriptions, 10, 1000)


class Demo:
    def __init__(self, device='cpu', prompt=None):
        self.device = device
        self.prompt = '''You are given a logic puzzle from the RAVEN dataset. The first shape on the first row is {}, the second shape on the first row is {}, the third item on the first row is {}. The first shape on the second row is {}, the second shape on the second row is a {}, the third shape on the second row is {}. The first shape on the third row is {}, the second shape is {}. Based on this, what is the third shape on the third row? You can only choose between: {}, {}, {}, {}, {}, {}, {}, {}'''

    def load_VLM(self, model):
        print(f'loading {model}...')
        self.VLM = CLIP(model, self.device)
        
        # Templates for predictions
        self.template_type = ['The figure is shaped as a {}'.format(value) for value in SHAPES.values()]
        self.template_color = ['The color of the figure is {}'.format(value) for value in COLORS.values()]
        self.template_angle = ['The figure is rotated by {} degrees'.format(value) for value in ANGLES.values()]
        self.template_sizes = ['The percentage of the image covered by the figure is {}'.format(value) for value in SIZES.values()]
        
    def load_LM(self, model, model_class=AutoModelForSeq2SeqLM):
        print(f'loading {model}...')
        self.LM = LM(model, model_class, self.device)

    def get_attributes(self, puzzle):
        answers = []
                
        for c_image in puzzle:
            angle = (self.VLM.forward(c_image, self.template_angle)).split()[-2]
            color = (self.VLM.forward(c_image, self.template_color)).split()[-1]
            size = (self.VLM.forward(c_image, self.template_sizes)).split()[-1]
            type = (self.VLM.forward(c_image, self.template_type)).split()[-1]
            
            answers.append([angle, color, size, type])
            
        return answers
    
    def OpenCV_pred_attributes(self):
        answers = []
        opencvmodel = OpenCV()
        
        for i in range(self.test_set.len()):
            puzzle = self.test_set.get_puzzle(i)
            puzzle_answers = []
            
            for c_image in puzzle:
                puzzle_answers.append(opencvmodel.detect_shape(c_image))
                
            answers.append(puzzle_answers)
            
        return answers
            
    def generate_prompts(self, attributes):
        return self.prompt.format(*[generate_single_description(attr) for attr in attributes])

    def get_descriptions(self, attributes):
        descriptions = []
        for puzzle in attributes:
            shape_desc = []
            for shape in puzzle[-8:]:
                for angle, color, size, type in shape:
                    shape_desc.append(generate_single_description(angle, color, size, type))
            descriptions.append(shape_desc)
        return descriptions

    def inference(self, prompt):
        pred = self.LM.forward(prompt)
        
        return pred

    def forward(self, puzzle):
        attributes = self.get_attributes(puzzle)
        prompt = self.generate_prompts(attributes)
        pred = self.inference([prompt])
        return pred
    
    
def main():
    SM = Demo()
    file = open("predictions.txt", "a")  # append mode
    
    for i in range(7000):
        puzzle = SM.test_set.get_puzzle(i)
        pred = SM.forward(puzzle)
        print(pred)
        file.write("{}\n".format(pred))
        
    file.close()

if __name__ == '__main__':
    main()
