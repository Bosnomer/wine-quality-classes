import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
from lime import lime_tabular
import pickle


dataset_path = "dataset/winequality-red.csv"

class Model:
    def __init__(self, datafile = dataset_path):
        self.df = pd.read_csv(datafile)
        category = pd.cut(self.df.quality,bins=[0,5,10,],labels=['Bad','Good'])
        self.df.insert(12,'Result',category)
        self.random_forest = RandomForestClassifier()
        
    def split(self, test_size):
        X = np.array(self.df.drop(['Result', 'quality'], axis=1))
        y = np.array(self.df['Result'])
        #print(X[0:1], y[0]) 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    def fit(self):
        self.model = RandomForestClassifier(random_state= 50).fit(self.X_train, self.y_train)
        pickle.dump(self.model, open('model/model.pkl', 'wb'))
    
    def explainability(self):
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.X_train),
            feature_names=self.df.drop(['Result', 'quality'], axis=1).columns,
            class_names=['Bad', 'Good'],
            mode='classification')
        self.X_test_df = pd.DataFrame(self.X_test, columns = self.df.drop(['Result', 'quality'], axis=1).columns)
        exp = explainer.explain_instance(
            data_row=self.X_test_df.iloc[18],
            predict_fn=self.model.predict_proba
		)
        exp.save_to_file('explainability/stats.html')
    
    def predict(self, input_value):
        result = self.model.predict(input_value)
        return result[0]

if __name__ == '__main__':
    model_instance = Model()
    model_instance.split(0.2)
    model_instance.fit()    
    model_instance.explainability()
    print("Accuracy score of model: ", model_instance.model.score(model_instance.X_test, model_instance.y_test))
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
    
    print("Quality Prediction: ", model_instance.predict([lst]))
