import numpy as np
from typing import Union, Dict
import pandas as pd

def fake_data() -> pd.DataFrame:
    data = {
    "Feature 1": [5.1, 4.9, 5.0, 6.5, 6.3, 6.6, 7.2, 6.9, 6.8, 5.5, 5.7, 5.8, 7.3, 7.1, 7.0],
    "Feature 2": [3.5, 3.0, 3.4, 3.0, 2.9, 3.0, 3.2, 3.1, 3.0, 2.4, 2.6, 2.7, 2.9, 3.0, 3.1],
    "Feature 3": [1.4, 1.4, 1.5, 4.5, 4.3, 4.6, 6.0, 5.4, 5.5, 3.8, 3.5, 3.9, 6.3, 5.9, 5.8],
    "Feature 4": [0.2, 0.2, 0.2, 1.5, 1.3, 1.4, 2.3, 2.1, 2.2, 1.1, 1.0, 1.2, 2.4, 2.1, 2.2],
    "Feature 5": [0.5, 0.6, 0.4, 1.2, 1.1, 1.3, 2.5, 2.3, 2.4, 0.9, 1.0, 1.2, 2.8, 2.6, 2.7],
    "Class": [0, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 0, 2, 0]
    }

    df = pd.DataFrame(data) 

    return df
    
class GaussianNB:

    def __init__(self, df:pd.DataFrame):
        self.df = df
        #This dict holds all the neccessary statistics 
        self.stats = {}

    def fit(self):
        
        total = self.df.shape[0]
        
        for class_vals in self.df['Class'].unique():

            class_df = self.df[self.df['Class'] == class_vals ]
            class_df = class_df.drop("Class",axis=1)

            class_prob = class_df.shape[0] / total

            mean_values = []
            std_vals = []
            
            for features in class_df.columns:
                feature_list = class_df[features].to_numpy()
                mean = np.mean(feature_list)
                std_dv = np.std(feature_list)
                
                mean_values.append(mean)
                std_vals.append(std_dv)
                # feature_vals.append(    
                #     {
                #         features:{
                #         "mean":mean,
                #         "std_dv":std_dv
                #     }
                #     }
                # )

            self.stats[class_vals] = {
                'class_prob':class_prob,
                'mean_matrix':mean_values,
                'std_matrix': std_vals
            }
            
    def _calculate_pdf(self,value:Union[int,float],mean:Union[int,float],std:Union[int,float]) -> float:
        
        exponent = np.exp((-(value - mean) ** 2) / (2 * std ** 2))
        coef = 1 / (np.sqrt(2 * np.pi * std ** 2))
        
        # print(value,mean,std)
        # print(exponent,coef,coef * exponent)
        # print("\n")
        return coef * exponent
        
    def predict(self,feature_array:np.array) -> Dict:
        
        assert len(feature_array) == (self.df.shape[1] - 1), "Given input is not of same size as training features"
        
        final_prob = {}
        
        for class_vals in self.stats:
            
            mean_array = self.stats[class_vals]['mean_matrix']
            std_matrix = self.stats[class_vals]['std_matrix']
            class_probablity = np.log(self.stats[class_vals]['class_prob'])
            
            for idx,val in enumerate(feature_array):
                
                mean_val = mean_array[idx]
                std_val = std_matrix[idx]
                
                class_probablity += np.log(self._calculate_pdf(value=val,mean=mean_val,std=std_val))
                
            final_prob[class_vals] = class_probablity
        
        return final_prob
        

if __name__ in '__main__':
    df = fake_data()
    gnb = GaussianNB(df=df)
    gnb.fit()
    
    new_entry = np.array([0.3,0.3,0.5,0.5,0.5])
    pred = gnb.predict(feature_array=new_entry)
    
    print(pred)