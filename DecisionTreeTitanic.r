######################################### R program to fit a decision tree, and carry out cross validation #########################################

######################################### C2 P1 Import needed packages for math and machine decision trees #########################################

#Needed to create classfication function
#install.packages("rpart", dep=TRUE)
library(rpart)

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = read.csv(train_url, as.is=TRUE)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = read.csv(test_url, as.is=TRUE)

#Define a function to clean up the data and add variables to be used for modelling

PreProcessingBeforeModel <- function(train){
    #For any NA values of Age, set them to the median
    train$Age[is.na(train$Age)] <- rep(median(train$Age[!is.na(train$Age)]), sum(is.na(train$Age)))
    
    # Impute the missing value with the median
    train$Fare[is.na(train$Fare)] <- rep(median(train$Fare[!is.na(train$Fare)]), sum(is.na(train$Fare)))
    
    # Create the column Child and assign to 'NaN'
    train$Child = NA
    
    # Assign 1 to passengers under 18, 0 to those 18 or older.
    train$Child[train$Age < 18] <- rep(1, sum(train$Age < 18))
    train$Child[train$Age >= 18] <- rep(0, sum(train$Age >= 18))
    
    #Convert predictor variables to format that is readable by classifier, testing to see if needed for R's decision tree
    #Convert the male and female groups to integer form
    #train["Sex"][train["Sex"] == "male"] = 0
    #train["Sex"][train["Sex"] == "female"] = 1
    
    #Imput the Embarked variable with "S" if there are NA values.
    train$Embarked[is.na(train$Embarked)] = rep("S", sum(is.na(train$Embarked)))
    
    #Convert the Embarked classes to integer form, testing to see if needed for R's decision tree
    #train["Embarked"][train["Embarked"] == "S"] = 0
    #train["Embarked"][train["Embarked"] == "C"] = 1
    #train["Embarked"][train["Embarked"] == "Q"] = 2
    return(train)
}


#Clean up training data
train = PreProcessingBeforeModel(train)


# Create dataset used for fitting the model
target_names = "Survived"
feature_names = c("Pclass", "Sex", "Age", "Fare")
train_fit = train[,c(target_names, feature_names)]

# Fit a decision tree: my_tree_one
Model_char <- paste("rpart(", target_names, " ~ ., method='class', data=train_fit, control=rpart.control(minsplit=2, cp=0.001))", sep="")
my_tree_one <- eval(parse(text=Model_char))

#Get accuracy of predictions on training dataset
sum(predict(my_tree_one, type="class") == train_fit[,target_names])/dim(train)[1]

#Carry out cross validation to get a more accurate picture of how the model will perform for future predictions. The cross validation result turns out to be about  20% lower than accuracy based solely on the training data. This indicates that the model is likely overfit to the training data.

#Function that repeats k times: randomly splits data into K folds, fits model on k-1 folds and tests accuracy of fitted data against the kth training set
#Accuracy is based upon successful prediction classifications, function would need to be enhanced to cover cover non classification models
K_Fold_CrossValidation_EJR <- function(train_fit, target_names, Model_char, k){


    # sample from 1 to k, nrow times (the number of observations in the data)
    train_fit$id <- sample(1:k, nrow(train_fit), replace = TRUE)
    list_folds <- 1:k


    prediction <- c()

    for (i in 1:k){
      # remove rows with id i from dataframe to create training set
      trainingset <- subset(train_fit, id %in% list_folds[-i])
      
      # select rows with id i to create test set
      testset <- subset(train_fit, id %in% c(i))
      
      #Rename training set data so don't have to recall model build function
      train_fit_temp <- train_fit
      train_fit <- trainingset
      
      #run a random forest model, eval parse changes normal text into executions. This is so Code that has strange formats from multiple sources can be called upon multiple times without rewriting
      mymodel <- eval(parse(text=Model_char))
      
      train_fit <- train_fit_temp
      rm(train_fit_temp)
                                                         
      # make prediction and calculate the accurary on the testing data
      temp <- sum(predict(mymodel, testset[,!is.element(names(testset), target_names)], type="class") == testset[,c(target_names)])/dim(testset)[1]

      # append this iteration's predictions to the end of the prediction data frame
      prediction <- c(prediction, temp)
      
    }

    return(mean(prediction))
}

K_Fold_CrossValidation_EJR(train_fit, target_names, Model_char, 10)


#Prune the tree in order to reduce overfitting
#Control overfitting by setting 
# Fit a decision tree: my_tree_one
Model_char <- paste("rpart(", target_names, " ~ ., method='class', data=train_fit, control=rpart.control(minsplit=5, cp=0.001))", sep="")
my_tree_one <- eval(parse(text=Model_char))

#Get accuracy of predictions on training dataset
sum(predict(my_tree_one, type="class") == train_fit[,target_names])/dim(train)[1]

#Carry out cross validation to get a more accurate picture of how the model will perform for future predictions. The cross validation result turns out to be about  11% lower than accuracy based solely on the training data. This indicates that the model is not overfitting as badly.
K_Fold_CrossValidation_EJR(train_fit, target_names, Model_char, 10)


# Simple implementation of the (normalized) gini score in numpy
# Fully vectorized, no loops,

y_true <- train_fit[,target_names]
y_pred <- as.numeric(as.character(predict(my_tree_one, type="class")))

#write.csv(data.frame(y_true, y_pred), file="Output/Gini_0610")

Gini_EJR <- function(y_true, y_pred){
    #get number of samples
    n_samples = length(y_true)
    
    # sort rows on prediction column 
    # (from largest to smallest)
    true_order <- y_true[order(y_true, decreasing=TRUE)]
    pred_order <- y_true[order(y_pred, decreasing=TRUE)]

    
    #arr = t(matrix(y_true, y_pred))
    #true_order = arr[arr[:,0].argsort()][::-1,0]
    #pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    
    
    # get Lorenz curves
    L_true = cumsum(true_order) / sum(true_order)
    L_pred = cumsum(pred_order) / sum(pred_order)
    L_ones = seq(1/n_samples, 1, length.out=n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = sum(L_ones - L_true)
    G_pred = sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return(G_pred/G_true)
}

Gini_EJR(y_true, y_pred)


# plot tree
plot(my_tree_one, uniform=TRUE, main="Classification Tree for Titanic")
text(my_tree_one, use.n=TRUE, all=TRUE, cex=.8)

#Clean up testing data
test = PreProcessingBeforeModel(test)

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[,c("Pclass", "Sex", "Age", "Fare")]

# Make your prediction using the test set
my_prediction = predict(my_tree_one, test_features, type='class')

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = test[,"PassengerId"]
my_solution = data.frame(Survived=my_prediction, row.names=PassengerId)

# Write your solution to a csv file with the name my_solution.csv
write.csv(my_solution, "Output/my_solution_one.csv")





