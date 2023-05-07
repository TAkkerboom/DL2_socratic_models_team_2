from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

import numpy as np
from os import listdir
import os
from sklearn.metrics import classification_report
import xml.etree.ElementTree as ET
import datetime
import torch
import torch.nn.functional as F
import torch.cuda

import requests # for APIs

from LM_research import Raven, save_results, main

import json


class API:
    
    def __init__(self, url="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-1-pythia-12b", *args):
        self.API_URL = url
        self.API_TOKEN = "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" # DO NOT POST THIS PUBLICLY
        self.headers = {"Authorization": f"Bearer {self.API_TOKEN}"}

    def forward(self, prompt):
        payload = {"inputs": "<|prompter|>" + prompt + "<|endoftext|><|assistant|>", "return_full_text":False}
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()

 

def inference(dataset, model, batch, samples):
    predictions = []
    j = 0
    for description, prompt in zip(dataset.descriptions[:samples], dataset.prompts[:samples]):
        output = model.forward(prompt)
        if output in description[8:]:
            pred = description[8:].index(output)
        else:
            pred = 8
        predictions.append(pred)
        j += 1
        if j % 10 == 0:
            print('currently at ', j * 10)
            
    return predictions

def save_results(pred, dataset, samples, model):
    now = datetime.datetime.now()
    curr_time = now.strftime('%m%d%H%M')
    np.savez('res_' + model.rstrip("/") + curr_time, predictions=np.array(pred), targets=dataset.targets[:int(samples)])
    print('saved as res_' + curr_time)


if __name__ == '__main__':
    main(inference, 'openassistant', research_class=API)
    
