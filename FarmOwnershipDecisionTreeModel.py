# Install required packages
# !pip install pandas scikit-learn dtreeviz

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from dtreeviz.trees import dtreeviz

# Set the working directory to Lab09 folder
# Replace the path with your actual path
working_directory = "C:/Users/ual-laptop/Desktop/MIS545/Lab09"
riceFarms_path = "IndonesianRiceFarms.csv"
riceFarms_full_path = f"{working_directory}/{riceFarms_path}"

# Reading IndonesianRiceFarms.csv into a pandas DataFrame
riceFarms = pd.read_csv(riceFarms_full_path)

# Displaying riceFarms in the console
print(riceFarms)

# Displaying the structure of riceFarms in the console
print(riceFarms.info())

# Displaying the summary of riceFarms in the console
print(riceFarms.describe())

# Randomly splitting the dataset into riceFarmsTraining (75% of records) 
# and riceFarmsTesting (25% of records) using 370 as the random seed
riceFarmsTraining, riceFarmsTesting = train_test_split(riceFarms, test_size=0.25, random_state=370)

# Generating the decision tree model to predict FarmOwnership based on the 
# other variables in the dataset. Use 0.01 as the complexity parameter.
riceFarmsDecisionTreeModel = DecisionTreeClassifier(ccp_alpha=0.01)
riceFarmsDecisionTreeModel.fit(riceFarmsTraining.drop(columns=['FarmOwnership']), riceFarmsTraining['FarmOwnership'])

# Displaying the decision tree visualization in Python
viz = dtreeviz(riceFarmsDecisionTreeModel, 
               riceFarmsTraining.drop(columns=['FarmOwnership']),
               riceFarmsTraining['FarmOwnership'],
               target_name='FarmOwnership',
               feature_names=list(riceFarms.columns[:-1]))
viz.view()

# Predicting classes for each record in the testing dataset and 
# storing them in riceFarmsPrediction
riceFarmPredictions = riceFarmsDecisionTreeModel.predict(riceFarmsTesting.drop(columns=['FarmOwnership']))

# Displaying riceFarmsPrediction on the console
print(riceFarmPredictions)

# Evaluating the model by forming a confusion matrix
riceFarmsConfusionMatrix = confusion_matrix(riceFarmsTesting['FarmOwnership'], riceFarmPredictions)

# Displaying the confusion matrix on the console
print(riceFarmsConfusionMatrix)

# Calculating the model predictive accuracy and store it into a variable 
# called predictiveAccuracy
predictiveAccuracy = accuracy_score(riceFarmsTesting['FarmOwnership'], riceFarmPredictions)

# Displaying the predictive accuracy on the console
print(predictiveAccuracy)

# Creating a new decision tree model using 0.007 as the complexity parameter
riceFarmsDecisionTreeModel2 = DecisionTreeClassifier(ccp_alpha=0.007)
riceFarmsDecisionTreeModel2.fit(riceFarmsTraining.drop(columns=['FarmOwnership']), riceFarmsTraining['FarmOwnership'])

# displaying the new decision tree visualization
viz2 = dtreeviz(riceFarmsDecisionTreeModel2, 
                riceFarmsTraining.drop(columns=['FarmOwnership']),
                riceFarmsTraining['FarmOwnership'],
                target_name='FarmOwnership',
                feature_names=list(riceFarms.columns[:-1]))
viz2.view()

# predicting classes for new decision tree
riceFarmPredictions2 = riceFarmsDecisionTreeModel2.predict(riceFarmsTesting.drop(columns=['FarmOwnership']))

# calculating its predictive accuracy using confusion matrix
riceFarmsConfusionMatrix2 = confusion_matrix(riceFarmsTesting['FarmOwnership'], riceFarmPredictions2)

predictiveAccuracy2 = accuracy_score(riceFarmsTesting['FarmOwnership'], riceFarmPredictions2)

# displaying the new predictive accuracy
print(predictiveAccuracy2)
