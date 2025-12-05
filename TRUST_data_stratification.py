
## TRUST SCORE FOR data stratification ###
## Input : excel file Test_cosine_cifar10.csv file with Trust score, Original class and Predicted class 
## Test_cosine_cifar10.csv contains values based on the PreactResNet18 model on CIFAR-10 dataset
import pandas as pd
import numpy as np

dataset = "cifar10"

print ("DATASET!!!!",dataset)

df = pd.read_csv(f'Test_cosine_{dataset}.csv')

## stratifications acros subset of points
percentage_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

accs_each = []
tot_pred = df.shape[0] ##number of rows

org_eq_pred  = df[(df['Org class']==df['Pred class'])].shape[0]

print ("acc cosine",org_eq_pred/tot_pred)
for per in percentage_list:
        per_data_to_consider = int(tot_pred*per) 
        sort_tot_pred = df.sort_values('Trust') 
        df1 = sort_tot_pred.tail(per_data_to_consider) 
        org_eq_pred  = df1[(df1['Org class']==df1['Pred class'])].shape[0]
        acc = org_eq_pred/per_data_to_consider            
        accs_each +=[acc]

accs_each = [i * 100 for i in accs_each]
print (percentage_list)
print (accs_each)

