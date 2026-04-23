# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model Name: Income classifier
Model Type: Binary classification
Algorithm: RandomRofrestClassifier
Version: 1.0
Framwork: scikit_learn
Deployment: FastAPI
Input: Census data
Output: Income <=50K or >50K

## Intended Use

This model is designed to predict whether the income of a persion is larger than 50K or less than/ equal to 50K

## Training Data

Traning data is census data with 
Categirical features: 
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

and Numerical features:
- age
- fnlgt
- education-num
- capital-gain
- capital-loss
- hours-per-week

Target data is salary

## Evaluation Data

he evaluation data is a held-out test split from the original dataset
80% Training data and 20% Test data

## Metrics

precision: 0.95
recall: 0.92
fbeta: 0.93

## Ethical Considerations

This dataset contains sensitive demographic attributes such as race, sex, and native country.

## Caveats and Recommendations

The predict is not 100%, only one time predict may not be the right answer
The Input data shoud be the same structure as training data