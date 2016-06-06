######################################### C2 P1 Import needed packages for math and machine decision trees #########################################
# Import the Pandas library
import pandas as pd

#Import the Numpy library
import numpy as np

#Import 'tree' from scikit-learn library
from sklearn import tree
from sklearn.tree import export_graphviz

#Needed for graphing decision tree
from sklearn.externals.six import StringIO  
import pydotplus 

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)


#Define a function to clean up the data and add variables to be used for modelling

def PreProcessingBeforeModel(train):
    #For any NA values of Age, set them to the median
    train["Age"] = train["Age"].fillna(train["Age"].median())
    
    # Impute the missing value with the median
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())
    
    # Create the column Child and assign to 'NaN'
    train["Child"] = float('NaN')
    
    # Assign 1 to passengers under 18, 0 to those 18 or older.
    train["Child"][train["Age"] < 18] = 1
    train["Child"][train["Age"] >= 18] = 0
    
    #Convert predictor variables to format that is readable by classifier
    #Convert the male and female groups to integer form
    train["Sex"][train["Sex"] == "male"] = 0
    train["Sex"][train["Sex"] == "female"] = 1
    
    #Impute the Embarked variable
    train["Embarked"] = train["Embarked"].fillna("S")
    
    #Convert the Embarked classes to integer form
    train["Embarked"][train["Embarked"] == "S"] = 0
    train["Embarked"][train["Embarked"] == "C"] = 1
    train["Embarked"][train["Embarked"] == "Q"] = 2
    return train


#Clean up training data
train = PreProcessingBeforeModel(train)


# Create the target and features numpy arrays: target, features_one
target_names = "Survived"
target = train[target_names].values
feature_names = ["Pclass", "Sex", "Age", "Fare"]
features_one = train[feature_names].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

#Visualize decision tree
dot_data = StringIO() 
tree.export_graphviz(my_tree_one, out_file=dot_data, feature_names=feature_names, class_names=target_names, filled=True, rounded=True,  special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("Titanic.pdf") 

#Clean up testing data
test = PreProcessingBeforeModel(test)

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("Output/my_solution_one.csv", index_label = ["PassengerId"])


















