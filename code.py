#importing all dependecies

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
#=====================================================#

card_df = pd.read_csv('card_transdata.csv')

card_df.isnull().sum() #how many null values each feature.

fraud_df = card_df.value_counts(card_df['fraud']).rename(index = 'Number of cases').to_frame() # original distribution for each operation.
print(fraud_df)

#Visual raw data distribution 
colors = ['slategrey','lightpink']
fraud_df.plot(kind='pie',subplots = 'True', autopct='%1.0f%%',colors=colors,labels = {'Legit':0,'Fraud':1},ylabel='',figsize=(3.1,3.1))

#separating for resampling 
legit = card_df[card_df.fraud == 0]
fraud = card_df[card_df.fraud == 1]

legit_sample = legit.sample(n = 87403) #n is the total of fraudulent operations. 
sample = pd.concat( [legit_sample, fraud] , axis = 0) # join the dataframes equally distributed.

sample.to_csv('model_data.csv',index=False) #Optionally, I decided to save this organized data frame into a csv file for further analysis. 

#Standardization 
ind_var = sample.drop(columns='fraud',axis=1) # Only independent variables
transform = preprocessing.StandardScaler()  # performs the standardization.
X = transform.fit_transform(ind_var) #Independent variables should go on the x-axis.

Y = sample['fraud'].to_numpy() #Target value

#Split the data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, stratify = Y, random_state = 2)

model = LogisticRegression()
model.fit(X_train,Y_train) #training the model

model.get_params(deep=True) #show the model parameters

#Model Evaluation

#Training accuracy
Y_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(Y_train_prediction, Y_train)
print("Training Accuracy: %.3f " % training_accuracy)

#Testing accuracy
Y_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(Y_test_prediction, Y_test)
print("Test Accuracy: %.3f " % testing_accuracy)

#Confusion Matrix
cm = confusion_matrix(Y_test,Y_test_prediction, labels = model.classes_)
displayCM = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['legit','fraud'])
displayCM.plot(cmap='Purples') #plot the confusion matrix changing the color scale


