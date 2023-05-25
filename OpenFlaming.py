from src.dataset import Raven
from src.model import VLM, LM, OpenCV
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
import requests
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from sklearn.metrics import classification_report

class Flamingo:

    def __init__(self,device):
        self.PATH = "/home/lcur1621/DL2_socratic_models_team_2/RAVEN-10000"
        self.device = device
        self.prompt = '''You are given a logic puzzle from the RAVEN dataset. The puzzle looks like <image>.
        Based on this image, what is the third shape on the third row?<|endofchunk|> You can only choose between the following shapes:
        Shape 1: <image><|endofchunk|>, Shape 2: <image><|endofchunk|>, Shape 3: <image><|endofchunk|>, Shape 4: <image><|endofchunk|>,
        Shape 5: <image><|endofchunk|>, Shape 6: <image><|endofchunk|>, Shape 7: <image><|endofchunk|>, Shape 8: <image><|endofchunk|>. The answer of the RAVEN puzzle logic puzzle is shape '''
        self.flamingomodel, self.image_processor, self.tokenizer = create_model_and_transforms(
                                                            clip_vision_encoder_path="ViT-L-14",
                                                            clip_vision_encoder_pretrained="openai",
                                                            lang_encoder_path="/home/lcur1621/DL2_socratic_models_team_2/LLamaHugginface/hugginfacellama",
                                                            tokenizer_path="/home/lcur1621/DL2_socratic_models_team_2/LLamaHugginface/hugginfacellama",
                                                            cross_attn_every_n_layers=4,
                                                            inference=True,
                                                            precision = 'fp16',
                                                            device = self.device,
                                                            checkpoint_path=hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt",token = "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")
                                                            )
        self.flamingomodel.to(self.device)
    
    def dataset(self, fig_type):
        self.fig_type = fig_type
        self.test_set = Raven(self.PATH, "", self.fig_type)
        self.test_set.load_data()
        print(str(self.test_set.len()))

    def Preprocess(self,puzzle):
        self.vision_x=[]
        for i in puzzle:
            self.vision_x.append(self.image_processor(i).unsqueeze(0).to(torch.float16).to(self.device))
        self.vision_x = torch.cat(self.vision_x, dim=0)
        self.vision_x = self.vision_x.unsqueeze(1).unsqueeze(0)       

    def TokenizerFlamingo(self):
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        self.lang_x = self.tokenizer(
            [self.prompt],
            return_tensors="pt",
        )
        self.lang_x.to(torch.float16)
        self.lang_x.to(self.device)
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
        print("Doing inference")
        for i in range(self.test_set.len()):
            print(f"Loading puzzle:{i}")
            puzzleanswers = self.test_set.get_puzzle(i)[8:]
            completepuzzle = Image.fromarray(self.test_set.items[i].grid).convert("P")
            puzzleanswers = [completepuzzle]+puzzleanswers
            self.Preprocess(puzzleanswers)
            self.TokenizerFlamingo()
            self.InferenceFlamingo()

            prediction =self.tokenizer.decode(self.generated_text[0])
            print(f"The prediction is:{prediction}")
            targets.append(self.test_set.items[i].target_id)
            predictions.append(prediction)
         
        return predictions, targets

    def forward(self):
        prediction, targets = self.inference()
        np.savez('Flamingo'+self.fig_type, predictions=np.array(prediction), targets=np.array(targets))
        # print(classification_report(targets, predictionnums, labels=list(range(0, 8))))



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    fig_types = ['center_single']
    # 'center_single',
    FlamingoModel = Flamingo(device)
    for fig_type in fig_types:
        FlamingoModel.dataset(fig_type)
        FlamingoModel.forward()
