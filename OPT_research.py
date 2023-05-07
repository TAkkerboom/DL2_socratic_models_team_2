from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, OPTForCausalLM, OPTModel, OPTForQuestionAnswering

import numpy as np
from os import listdir
import os
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET
import datetime
import torch
import torch.nn.functional as F
import torch.cuda
import logging
import argparse

import requests # for APIs

from LM_research import Raven, save_results, main


class OPT:
    
    def __init__(self, model, device, model_class):
        # logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        self.model = model_class.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.model.to(device)
        
    def forward(self, prompt):
        # inputs = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
        # inputs = {key: value.to(self.device) for key, value in self.tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True).items()}
        # outputs = self.model(**inputs)
        torch.manual_seed(4)
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        # text, question = prompt.split('Based')
        inputs = self.tokenizer(question, text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        answer_offset = len(self.tokenizer.encode(question))

        predict_answer_tokens = inputs.input_ids[
            0 + answer_start_index : answer_end_index + 1
        ]
        predicted = self.tokenizer.decode(predict_answer_tokens)
        return predicted


def inference(dataset, model, batch_size, samples):
    predictions = []
    j = 0
    # prompts = create_batches(dataset.prompts, batch_size)
    # descriptions = create_batches(dataset.descriptions, batch_size)
    # N  = int(samples / batch_size)
    for description, prompt in zip(dataset.descriptions, dataset.prompts):
        out = model.forward(prompt)
    
        print(out)
        if out in description[8:]:
            pred = description[8:].index(out)
        else:
            pred = 8
        predictions.append(pred)
        j += 1
        if j % 1 == 10:
            print('currently at ', j * 10)
            
    return predictions

def save_results(pred, dataset, samples, model):
    now = datetime.datetime.now()
    curr_time = now.strftime('%m%d%H%M')
    np.savez('res_' + model.rstrip("/") + curr_time, predictions=np.array(pred), targets=dataset.targets[:int(samples)])
    print('saved as res_' + curr_time)
 
      
if __name__ == '__main__':
    main(infer=inference, mod='distilbert', research_class=OPT)
    
