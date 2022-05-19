import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
from lime import lime_tabular
import pickle

model_path = 'model/model.pkl'
# model = pickle.load(open(model_path,'rb'))

class Model_Predict:
	def __init__(self, modelfile=model_path):
		self.model = pickle.load(open(modelfile,'rb'))

	def pred(self, values):
		result = self.model.predict(values)
		return result[0]

if __name__ == '__main__':
    model_instance = Model_Predict()

    print("Enter values of:")
    fa = float(input("Enter Fixed Acidity: "))
    va = float(input("Enter Volatile Acidity: "))
    ca = float(input("Enter Citric Acid: "))
    rs = float(input("Enter Residual Sugar: "))
    ch = float(input("Enter Chlorides: "))
    fsd = float(input("Enter Free Sulphur dioxide: "))
    tsd = float(input("Enter Total Sulphur dioxide: "))
    den = float(input("Enter Density: "))
    ph = float(input("Enter pH value: "))
    sul = float(input("Enter Sulphates: "))
    alc = float(input("Enter Alcohol: "))
    lst = [fa,va,ca,rs,ch,fsd,tsd,den,ph,sul,alc]
    
    print("Quality Prediction: ", model_instance.pred([lst]))

