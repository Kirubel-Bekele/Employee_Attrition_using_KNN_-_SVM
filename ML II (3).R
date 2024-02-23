# L.J. Russell
# Kirubel Bekele


# 1.  Pre-process the data and prepare it for running the following classification models (KNN and SVM). (15 points)


# Load necessary libraries
library(fastDummies)
library(caret) # Provides functions for training and plotting machine learning models.
library(class) # Contains functions for k-Nearest Neighbors (kNN) classification.
library(dplyr) # Facilitates data manipulation and transformation.
library(pROC)  # Used for Receiver Operating Characteristic (ROC) analysis.
library(e1071)
# Read the dataset and convert it into a dataframe.
employee_data = read.csv("EmployeeData.csv", stringsAsFactors =  TRUE)


# Select only the relevant columns, excluding EmployeeID and StandardHours.
employee_data = employee_data %>% select(-c(EmployeeID, StandardHours))

# Convert specific columns to factor data type for classification analysis.
employee_data$Attrition <- as.factor(employee_data$Attrition)
employee_data$BusinessTravel <- as.factor(employee_data$BusinessTravel)
employee_data$Education <- as.factor(employee_data$Education)
employee_data$Gender <- as.factor(employee_data$Gender)
employee_data$MaritalStatus <- as.factor(employee_data$MaritalStatus)
employee_data$JobLevel <- as.factor(employee_data$JobLevel)
employee_data$EnvironmentSatisfaction <- as.factor(employee_data$EnvironmentSatisfaction)
employee_data$JobSatisfaction <- as.factor(employee_data$JobSatisfaction)

# Split data into training and test sets (70% training, 30% test).
set.seed(456) # Set seed for reproducibility
index = sample(nrow(employee_data), 0.7 * nrow(employee_data)) # Randomly sample indices for training set
train_data = employee_data[index,] # Create training set
test_data = employee_data[-index,] # Create test set

# Check the distribution of the target variable in both training and test sets.
prop.table(table(train_data$Attrition))
prop.table(table(test_data$Attrition))


# Ensure that the response variable is a factor
train_data$Attrition <- as.factor(train_data$Attrition)

#Resampling to address class imbalance (optional for this dataset)
train_data = upSample(x=train_data[,-2], y=train_data[,2], yname = "Attrition")

# Check for missing values in both training and test sets.
sapply(train_data, function(x){sum(is.na(x))})
sapply(test_data, function(x){sum(is.na(x))})

# Calculate the fraction of missing data in specific columns of training and test sets.
sum(is.na(train_data$NumCompaniesWorked))/nrow(train_data)
sum(is.na(train_data$JobSatisfaction))/nrow(train_data)
sum(is.na(train_data$TotalWorkingYears))/nrow(train_data)
sum(is.na(train_data$EnvironmentSatisfaction))/nrow(train_data)
sum(is.na(test_data$NumCompaniesWorked))/nrow(test_data)
sum(is.na(test_data$JobSatisfaction))/nrow(test_data)
sum(is.na(test_data$TotalWorkingYears))/nrow(test_data)
sum(is.na(test_data$EnvironmentSatisfaction))/nrow(test_data)

# The getMode function calculates the mode (most frequent value) of a given vector. It first identifies unique values, counts their occurrences, and returns the value that appears most often.
getMode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Impute missing numerical values with the median and categorical variables with the mode from the training data on train_data.
train_data$NumCompaniesWorked[is.na(train_data$NumCompaniesWorked)] <- mean(train_data$NumCompaniesWorked, na.rm = TRUE)
train_data$TotalWorkingYears[is.na(train_data$TotalWorkingYears)] <- mean(train_data$TotalWorkingYears, na.rm = TRUE)
train_data$EnvironmentSatisfaction[is.na(train_data$EnvironmentSatisfaction)] <- getMode(train_data$EnvironmentSatisfaction)
train_data$JobSatisfaction[is.na(train_data$JobSatisfaction)] <- getMode(train_data$JobSatisfaction)

#Impute missing numerical values with the median and categorical variables with the mode from the training data on test_data.
test_data$NumCompaniesWorked[is.na(test_data$NumCompaniesWorked)] <- mean(train_data$NumCompaniesWorked, na.rm = TRUE)
test_data$TotalWorkingYears[is.na(test_data$TotalWorkingYears)] <- mean(train_data$TotalWorkingYears, na.rm = TRUE)
test_data$EnvironmentSatisfaction[is.na(test_data$EnvironmentSatisfaction)] <- getMode(train_data$EnvironmentSatisfaction)
test_data$JobSatisfaction[is.na(test_data$JobSatisfaction)] <- getMode(train_data$JobSatisfaction)

# Confirm there are no missing values left after imputation.
sapply(train_data,function(x){sum(is.na(x))})
sapply(test_data, function(x){sum(is.na(x))})

# Apply dummy encoding to categorical variables in train_x and test_x using the fastDummies package.
train_data <- train_data %>% dummy_cols(select_columns = c("Education", "BusinessTravel", "Gender", "JobLevel", "MaritalStatus", "EnvironmentSatisfaction", "JobSatisfaction"), remove_first_dummy = T, remove_selected_columns = T)
test_data <- test_data %>% dummy_cols(select_columns = c("Education", "BusinessTravel", "Gender", "JobLevel", "MaritalStatus", "EnvironmentSatisfaction", "JobSatisfaction"), remove_first_dummy = T, remove_selected_columns = T)




# 2.  Run a K-Nearest Neighbor model to build a predictive model for employees' attrition. Try out different values of k to find an optimal model. (25 points). 


# Scale the numeric columns in the training and test data.
numeric_columns <- c('Age','DistanceFromHome','YearsAtCompany','YearsWithCurrManager','NumCompaniesWorked','TotalWorkingYears','Income','TrainingTimesLastYear') 
train_x <- train_data
# Remove Attrition column from train_x as it is the target variable.
train_x$Attrition <- NULL
train_x[, numeric_columns] <- scale(train_data[, numeric_columns])

test_x <- test_data
# Remove Attrition column from test_x as it is the target variable.
test_x$Attrition <- NULL
test_x[, numeric_columns] <- scale(test_data[, numeric_columns],
                                   center = apply(train_data[, numeric_columns], 2, mean), 
                                   scale = apply(train_data[, numeric_columns], 2, sd))


# Save the target variable (Attrition) in train_y and test_y.
train_y = train_data$Attrition
test_y = test_data$Attrition


# Determine an initial value for k in kNN using the square root of the number of rows in train_x.
k = sqrt(nrow(train_x))
k


# Train a kNN model with k=72.
set.seed(456)
model_knn = knn(train=train_x,
                test=test_x, 
                cl=train_y, k=72)
#evaluation
confusionMatrix(data = model_knn, 
                reference = test_y,
                positive = "Yes")
# Initializes a matrix to store k values and their corresponding AUC scores.
output = matrix(ncol=2, nrow=50)
# Create loop over k values from 1 to 50
for (k_val in 1:50){
  
  #Storing predicted values for each run of the loop (i.e., each value of k)
  set.seed(456)
  temp_pred = knn(train = train_x
                  ,test = test_x
                  ,cl = train_y
                  ,k = k_val)
  
  #Calculate performance measures for the given value of k
  temp_eval = confusionMatrix(temp_pred, test_y) 
  temp_acc = temp_eval$overall[1]
  
  #Add the calculated accuracy as a new row in the output matrix
  output[k_val, ] = c(k_val, temp_acc) 
}
#Convert the output to a data frame 
output = as.data.frame(output)
names(output) = c("K_value", "Accuracy")

#plot the accuracy and K-vlaues
ggplot(data=output, aes(x=K_value, y=Accuracy, group=1)) +
  geom_line(color="red")+
  geom_point()+
  theme_bw() 



# Train another kNN model with k=1.
set.seed(456)
model_1 = knn(train=train_x,
              test=test_x, cl=train_y, k=1)

# Obtain predicted probabilities for model_1.
set.seed(456)
model_1_probs = attributes(knn(train=train_x, 
                               test=test_x, 
                               cl=train_y, 
                               k=1, prob=TRUE))$prob
#evaluating model-1 performance
confusionMatrix(data = model_1,
                reference = test_y, 
                positive = "Yes")
# Evaluate model_1 using ROC and AUC.
auc_score1 <- roc(response = test_y, 
                  predictor = as.numeric(model_1_probs))
auc_score1

# Train another kNN model with k=1.
set.seed(456)
model_k2 = knn(train=train_x,
              test=test_x, cl=train_y, k=2)

# Obtain predicted probabilities for model_2.
set.seed(456)
model_k2_probs = attributes(knn(train=train_x, 
                               test=test_x, 
                               cl=train_y, 
                               k=2, prob=TRUE))$prob

# Evaluate model_1 using ROC and AUC.
auc_score2 <- roc(response = test_y, 
                  predictor = as.numeric(model_k2_probs))
auc_score2

# Compute confusion matrix for model_1.
confusionMatrix(data = model_k2,
                reference = test_y, 
                positive = "Yes")

#using Caret package
# Set up 10-fold cross-validation for kNN model training using the caret package.
control <- trainControl(method="cv", number=10)

# Train a kNN model using CARET with a range of k values from 1 to 15.
knn_cv <- train(
  Attrition ~ ., data=train_data, 
  method="knn",  trControl=control,
  preProcess = c("center","scale"), 
  tuneGrid = expand.grid(k = 1:15))
knn_cv

# Plot the cross-validation results to assess the performance across different k values.
plot(knn_cv)
plot(varImp(knn_cv),5)

# Predict the test data using the trained kNN model (knn_cv).
model_2 <- predict(knn_cv, newdata=test_data, type="raw")
model_2_probs <- predict(knn_cv, test_data, type="prob")[,2]

# Evaluate model_2 using a confusion matrix.
confusionMatrix(data = model_2, 
                reference = test_y, 
                positive = "Yes")


# Plotting ROC curves
plot.roc(test_y, model_1_probs, col = "blue", main = "ROC Curves", legacy.axes = TRUE)#this the model from k=1
plot.roc(test_y, model_k2_probs, col = "green", add = TRUE, legacy.axes = TRUE)#this is the model from k=2
plot.roc(test_y, model_2_probs, col = "red", lty = 2, add = TRUE)#the model form the caret package 

# Adding a legend
legend("bottomright", 
       legend = c("Model 1", "Model k2", "Model 2"), 
       col = c("blue", "green", "red"), 
       lty = c(1, 1, 2), 
       cex = 0.75)

# Calculate AUC scores for both models.
auc(test_y, model_1_probs)
auc(test_y, model_2_probs)



# 3. Run a Support Vector Machines model to build a predictive model for employees' attrition. Try out different model settings to find an optimal model. (25 points) 


# The following lines of code create and evaluate different Support Vector Machine (SVM) models using both the 'e1071' and 'caret' packages. These models are used for the classification task to predict employee attrition. Different SVM configurations, including linear and radial kernels, are explored. The models are evaluated using confusion matrices and ROC curves, and the area under the curve (AUC) is calculated for performance comparison.

# Support Vector Machine model with linear kernel
SVM1 = svm(formula = Attrition ~ ., 
           data= train_data,
           type = 'C-classification',
           kernel = 'linear',
           scale=TRUE) # Creates an SVM model with linear kernel using all variables for predicting Attrition.
summary(SVM1) # Summarizes the model's details.
head(SVM1$index) # Displays the first few indices of the model.

# Alternative SVM model with a different cost parameter
SVM2 = svm(formula = Attrition ~.,
           data = train_data,
           type = 'C-classification',
           kernel = 'linear',
           cost=0.1,
           scale=TRUE) # Creates another SVM model with a linear kernel and a cost parameter of 0.1.
summary(SVM2) # Summarizes the second SVM model's details.

# Tuning the SVM model for optimal parameters
set.seed(456) # Sets a seed for random number generation.
tune.out = tune(svm, Attrition ~., data = train_data, kernel = "linear", probability = TRUE, ranges = list(cost=c(0.001,0.01,0.1,1,5,10))) # Tunes SVM parameters (cost) to find the best model.
summary(tune.out) # Summarizes the tuning result.

# Selecting and applying the best tuned SVM model
bestmod = tune.out$best.model # Retrieves the best SVM model from the tuning process.
names(test_data) # Lists the column names of the test data.
SVM_pred=predict(bestmod, newdata=test_data[-2]) # Predicts Attrition on the test data using the best SVM model.
SVM_prob = attributes(predict(bestmod, newdata = test_data[-2],probability=TRUE))$probabilities[,2] # Computes predicted probabilities for the positive class.
confusionMatrix(data = SVM_pred,
                reference = test_data$Attrition,
                positive="Yes") # Generates a confusion matrix for the predictions.

# SVM with radial basis kernel
SVMR = svm(formula = Attrition ~ .,
           data = train_data,
           type = 'C-classification',
           cost=1,
           gamma=0.05,
           kernel='radial') # Creates an SVM model with radial basis kernel.
summary(SVMR) # Summarizes the radial SVM model's details.

# Tuning the radial SVM model
set.seed(456) # Sets a seed for random number generation.
tune.out2 = tune(svm, Attrition ~., data=train_data, kernel ="radial", probability=TRUE,
                 ranges=list(cost=c(0.01,0.1,1,5,10), gamma=c(0.5,1,2,3))) # Tunes radial SVM parameters (cost and gamma) to find the best model.
summary(tune.out2) # Summarizes the tuning result for the radial SVM.

# Applying the best tuned radial SVM model
bestmod2 = tune.out2$best.model # Retrieves the best radial SVM model from the tuning process.
SVMR_pred = predict(bestmod2, newdata = test_data[-2]) # Predicts Attrition on the test data using the best radial SVM model.
SVMR_prob = attributes(predict(bestmod2, newdata=test_data[-2], probability=TRUE))$probabilities[,2] # Computes predicted probabilities for the radial SVM model.

# Model evaluation for radial SVM
confusionMatrix(data = SVMR_pred, 
                reference = test_data$Attrition, 
                positive = "Yes") # Generates a confusion matrix for the radial SVM predictions.

# Copying training and test datasets for further use
train_data_copy = train_data # Creates a copy of the training data.
test_data_copy = test_data # Creates a copy of the test data.

# Relabeling levels in the copied datasets
levels(train_data_copy$Attrition) = c("No","Yes") # Relabels levels in 'Attrition' column for the training data copy.
levels(test_data_copy$Attrition) = c("No","Yes") # Relabels levels in 'Attrition' column for the test data copy.

# Setting up control parameters for training
ctrl = trainControl(method="cv",number=10,classProbs=TRUE) # Defines control parameters for cross-validation.

# Training a radial SVM model using caret package
set.seed(456) # Sets a seed for random number generation.
SVMR_caret = train(
  Attrition ~ ., data = train_data_copy,
  method = "svmRadial", trControl = ctrl, 
  preProcess = c("center","scale"), tuneLength = 10) # Trains a radial SVM model using the caret package.

# Model details and variable importance
SVMR_caret # Displays details of the trained radial SVM model.
plot(SVMR_caret) # Plots the performance of different configurations of the radial SVM model.
plot(varImp(SVMR_caret)) # Plots variable importance for the radial SVM model.

# Predicting and evaluating the model
SVMR_pred2 = predict(SVMR_caret, test_data_copy[,-2], type="raw") # Predicts Attrition on the test data copy using the caret SVM model.
SVMR_prob2 = predict(SVMR_caret, test_data_copy[,-2], type="prob")[,2] # Computes predicted probabilities for the positive class using the caret SVM model.

# Confusion matrix for the caret radial SVM model
confusionMatrix(data = SVMR_pred2, 
                reference = test_data_copy$Attrition, 
                positive = "Yes") # Generates a confusion matrix for the caret radial SVM predictions.

# ROC curve plotting
plot.roc(test_data$Attrition,SVM_prob,legacy.axes=T) # Plots ROC curve for the linear SVM model.
plot.roc(test_data$Attrition,SVMR_prob,add=TRUE,col="red",lty=2,legacy.axes=T) # Adds the ROC curve for the e1071 radial SVM model.
plot.roc(test_data_copy$Attrition,SVMR_prob2,add=TRUE,col="blue",lty=3) # Adds the ROC curve for the caret radial SVM model.
legend("bottomright",legend=c("SVM (linear)","SVM (radial)-e1071","SVM (radial)-caret"),
       col=c("black","red","blue"),lty=c(1,2,3),cex=0.75) # Adds a legend to the ROC plot.

# Calculating AUC for classifiers
auc(test_data$Attrition,SVM_prob) # Calculates the AUC for the linear SVM model.
auc(test_data$Attrition,SVMR_prob) # Calculates the AUC for the e1071 radial SVM model.
auc(test_data_copy$Attrition,SVMR_prob2) # Calculates the AUC for the caret radial SVM model.



# 4.  Compare the best KNN and SVM models by their model evaluation metrics. Which is a better model, and why? (10 points) 

#creating ROC object 
roc_obj_knn <- roc(test_y, model_2_probs)
roc_obj_svm <- roc(test_y, SVMR_prob)

#calculate auc for each model
auc_knn <- auc(roc_obj_knn)
auc_svm <- auc(roc_obj_svm)
#print the auc value 
cat("AUC for KNN Model:", auc_knn, "\n")
cat("AUC for SVM Model:", auc_svm, "\n")

#plot the ROC curve
plot(roc_obj_knn, col="blue", main="ROC Curves for KNN and SVM Models")
plot(roc_obj_svm, add=TRUE, col="red", lty=2)
legend("bottomright", legend=c("KNN.CV", "SVM (radial)-e1071"), col=c("blue", "red"), lty=c(1, 2), cex=0.75)



