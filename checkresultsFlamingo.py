from sklearn.metrics import classification_report
import os
import numpy as np
import re

class checkresults():
    def __init__(self, path):       
        self.path = path

    def results(self):
        npz = np.load(self.path, allow_pickle=True)
        prediction = npz["prediction"]
        targets = npz['targets']
        predictionlist = list(prediction)
        targetlist = list(targets)
        predictionnum = [prediction.split("The answer of the RAVEN puzzle logic puzzle is shape")[1]
        for prediction in predictionlist]
        finalprediction = []
        for pnum in predictionnum:
            finalprediction.append(int(''.join(filter(str.isdigit, pnum[:4])))-1)
        print(classification_report(list(targets), finalprediction, labels=list(range(0, 8))))

if __name__=='__main__':
    paths = os.listdir('./')
    pathnpz = []
    for path in paths:
        if path.endswith("center_single_correct.npz"):
            pathnpz.append(path)
    for npz in pathnpz:
        Flamingoresults = checkresults(path=npz)
        Flamingoresults.results()
