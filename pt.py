from LM_research import Raven, LM, create_batches
import numpy as np
from transformers import AutoModelForSeq2SeqLM
import torch


def inference(dataset, model, batch_size, samples):
    predictions = []
    j = 0
    prompts = create_batches(dataset['prompts'], batch_size)
    descriptions = create_batches(dataset['descriptions'], batch_size)
    N  = int(samples / batch_size)
    for description_batch, prompt_batch in zip(descriptions[:N], prompts[:N]):
        out = model.forward(prompt_batch.tolist())
        for output, description in zip(out, description_batch):
            description = description.tolist()
            if output in description[8:]:
                pred = description[8:].index(output)
            else:
                pred = 8
            predictions.append(pred)
        j += 1
        if j % 1 == 0:
            print('currently at ', j * 10)
            
    return predictions

def main(data_sets, dataset, infer=inference, mod='flanT5_xl', research_class=LM):
    models = {'flanT5_large': ["google/flan-t5-large", AutoModelForSeq2SeqLM],
              'flanT5_xl': ["google/flan-t5-xl", AutoModelForSeq2SeqLM],
              'flanT5_small': ["google/flan-t5-small", AutoModelForSeq2SeqLM],
              }

    # specify which model from models you want to select
    selected_model = models.get(mod, mod)
    model_version, model_class = selected_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('loading model {}...'.format(model_version))
    model = research_class(model_version, device, model_class)

    # data and model to device
    dataset.to_gpu(device)

    # starting inference loop with args
    print('running ... (this may take some time)')
    batch = 10
    samples = 100
    epochs = 100

    for i in range(epochs):
        template = model.forward([dataset.standard_prompt + ' This is a prompt for a large language model. Generate a variant on this prompt that may perform better.'])
        print(template)
        train_pred = infer(data_sets['train'], model, batch, samples)
        print('train')
        correct_pred = sum(np.array(train_pred) == data_sets['train']['targets'][:samples])
        print('train acc: ', correct_pred / samples)
        dev_pred = infer(data_sets['dev'], model, batch, samples)
        correct_pred = sum(np.array(dev_pred) == data_sets['dev']['targets'][:samples])
        print('valid acc: ', correct_pred / samples)
        break


if __name__ == '__main__':
    dataset = Raven()
    dataset.forward()

    targets = dataset.targets
    prompts = dataset.prompts
    attributes = dataset.attributes
    descriptions = dataset.descriptions


    # split the data into train, dev, and test sets 
    train_targets, dev_targets, test_targets = np.split(targets, [int(.6*len(targets)), int(.8*len(targets))]) 
    train_prompts, dev_prompts, test_prompts = np.split(prompts, [int(.6*len(prompts)), int(.8*len(prompts))]) 
    train_attributes, dev_attributes, test_attributes = np.split(attributes, [int(.6*len(attributes)), int(.8*len(attributes))]) 
    train_descriptions, dev_descriptions, test_descriptions = np.split(descriptions, [int(.6*len(descriptions)), int(.8*len(descriptions))])

    # Create object 
    data_sets = { 
        "train": {"targets": train_targets, "prompts": train_prompts, "descriptions": train_descriptions}, 
        "dev": {"targets": dev_targets, "prompts": dev_prompts, "descriptions": dev_descriptions}, 
        "test": {"targets": test_targets, "prompts": test_prompts, "descriptions": test_descriptions} 
    }
    main(data_sets, dataset, infer=inference, mod='flanT5_xl', research_class=LM)
