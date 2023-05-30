from sklearn.metrics import classification_report
import numpy as np

class checkresults():
    def __init__(self, pathroot):       
        self.path = pathroot

    def results(self):
        npz = np.load(self.path, allow_pickle=True)
        prediction = npz["predictions"]
        targets = npz['targets']
        predictionlist = list(prediction)
        predictionnum = [prediction.split("The answer of the RAVEN puzzle logic puzzle is shape")[1]
                        for prediction in predictionlist]
        finalprediction = []
        for pnum in predictionnum:
            finalprediction.append(int(''.join(filter(str.isdigit, pnum[:4])))-1)
        print(classification_report(list(targets), finalprediction, labels=list(range(0, 8))))
        print(predictionlist[3])

if __name__=='__main__':
    pathroot = "Result_Flamingo/Flamingocenter_single_complete.npz"
    Flamingoresults = checkresults(pathroot,)
    Flamingoresults.results()
