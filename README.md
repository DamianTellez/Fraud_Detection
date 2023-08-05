# Fraud Detection

![thumbnail](img/ecommerce-fraud-detection-and-prevention-thumbnail-1.png)

A machine learning model to detect suspicious credit card activities, which is a frequent issue companies have to deal with. 

## Database 

To address this problem, I used a public data set. The source was kept unrevealed due to privacy policies.

You can find the data in a csv file attached to this repository in the **database** folder. If it was helpful to you, you could give it a ⭐ 

## Model 

### Logistic Regression

This type of statistical model is used for classification and predictive analysis. Estimate the probability of a binary-outcome event occurring given a dataset of independent variables. 

For this study case, I used the following independent variables:

- distance from home
- distance from the last transaction
- median purchase price and current purchase price ratio
- repeat retailer (yes or not)
- used chip (yes or not)
- used pin number (yes or not)
- online order (yes or not)

### Model evaluation

Model accuracy: 94.0 %

### Model optimization

In this stage, I tuned the hyperparameters optimizing the model. By using **GridSearchCV** we find the best combination which result in the highest accuracy.

