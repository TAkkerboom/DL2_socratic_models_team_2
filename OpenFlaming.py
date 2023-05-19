from src.dataset import Raven
from src.model import VLM, LM, OpenCV
import numpy as np
import requests
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM
from PIL import Image
import requests
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from sklearn.metrics import classification_report
class Flamingo:

    def __init__(self,device):
        self.PATH = "RAVEN-10000"
        fig_types = ['center_single']

        self.test_set = Raven(self.PATH, 'test', fig_types[0])
        self.test_set.load_data()

        self.prompt = '''You are given a logic puzzle from the RAVEN dataset. The puzzle looks like <|endofchunk|><image>.
        Based on this image, what is the third shape on the third row? You can only choose between the following shapes:
        Shape 1: <|endofchunk|><image>, Shape 2: <|endofchunk|><image>, Shape 3: <|endofchunk|><image>, Shape 4: <|endofchunk|><image>,
        Shape 5: <|endofchunk|><image>, Shape 6: <|endofchunk|><image>, Shape 7: <|endofchunk|><image>, Shape 8: <|endofchunk|><image>'''
        self.flamingomodel, self.image_processor, self.tokenizer = create_model_and_transforms(
                                                            clip_vision_encoder_path="ViT-L-14",
                                                            clip_vision_encoder_pretrained="openai",
                                                            lang_encoder_path="hugginfacellama",
                                                            tokenizer_path="hugginfacellama",
                                                            cross_attn_every_n_layers=4
                                                        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
        self.flamingomodel.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.flamingomodel.to(device)


    def Preprocess(self,puzzle):
        self.vision_x=[]
        for i in puzzle:
            self.vision_x.append(self.image_processor(i).unsqueeze(0))
        self.vision_x = torch.cat(self.vision_x, dim=0)
        self.vision_x = self.vision_x.unsqueeze(1).unsqueeze(0)

    def TokenizerFlamingo(self):
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        self.lang_x = self.tokenizer(
            [self.prompt],
            return_tensors="pt",
        )
    
    def InferenceFlamingo(self):
        self.generated_text = self.flamingomodel.generate(
            vision_x=self.vision_x,
            lang_x=self.lang_x["input_ids"],
            attention_mask=self.lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
            )

    def inference(self):
        predictions = []
        targets = []
        for i in range(self.test_set.len()):
            puzzleanswers = self.test_set.get_puzzle(i)[8:]
            completepuzzle = Image.fromarray(self.test_set.items[i].grid).convert("P")
            puzzleanswers = [completepuzzle]+puzzleanswers
            self.Preprocess(puzzleanswers)
            self.TokenizerFlamingo()
            self.InferenceFlamingo()

            prediction =self.tokenizer.decode(self.generated_text[0])
            targets.append(self.test_set.get_answers(i))
            predictionnum = int(''.join(filter(str.isdigit, prediction)))-1
            predictions.append(predictionnum)
         
        return predictions, targets

    def forward(self):
        prediction, targets = self.inference()
        np.savez('Flamingo', predictions=np.array(prediction), targets=np.array(targets))
        print(classification_report(targets, prediction, labels=list(range(0, 8))))



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FlamingoModel = Flamingo(device)
    FlamingoModel.forward()
