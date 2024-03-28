import numpy as np
import pandas as pd
from sklearn.model_selection import KFold #train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import Preprocessing
from imblearn.over_sampling import RandomOverSampler
import time
import joblib
import os

start_time = time.time()

print("")
print("I'm here: Step 1!\n")

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')
preprocessor = Preprocessing() 

print("I'm here: Step 2!\n")

bin_factor=100
BH_powerspectra = preprocessor.array_collect(path_BH, bin_factor=bin_factor, BH=True)
NS_powerspectra = preprocessor.array_collect(path_NS, bin_factor=bin_factor, BH=False)
powerspectra=np.vstack((BH_powerspectra,NS_powerspectra))

print("I'm here: Step 3!\n")

X = powerspectra[:,0:2]
y = powerspectra[:,3]

print("I'm here: Step 4!\n")

post_processing_time = time.time()

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200,min_samples_leaf=20,min_samples_split=50)

print("I'm here: Step 5!\n")

# Perform k-fold cross-validation
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
mse_scores = []
accuracy_scores = []
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    rf_classifier.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf_classifier.predict(X_test)
    
    # Calculate and store the MSE
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)


print("I'm here: Step 6!\n")

# Calculate the average MSE across all folds
average_mse = np.mean(mse_scores)
average_accuracy = np.mean(accuracy_scores)

print("I'm here: Step 7!\n")

# Print the results
print(f'Average Mean Squared Error (MSE) across all folds: {average_mse}')
print()
print(f'Average Accuracy across all folds: {average_accuracy}')

i = 100
print()
print("Tiempo de procesamiento de archivos:", post_processing_time - start_time, "segundos")
print("Tiempo de entrenamiento con bineado", i, ":", time.time() - post_processing_time, "segundos")
#print("Tiempo de entrenamiento:", time.time() - post_processing_time, "segundos")

print()

model_info = {'model': rf_classifier,
              'average_mse': str(average_mse),
              'average_accuracy': str(average_accuracy),
              'rebin_factor':str(i),
              'reading_time': post_processing_time - start_time,
              'training_time':  time.time() - post_processing_time}

#joblib.dump(model_info, 'rf_classifier'+str(bin_factor)+'.pkl')
joblib.dump(model_info, 'rf_classifier'+str(i)+'.pkl')

# G added these lines
# Define a file path where you want to save the results
output_file = "results_PS_RF.txt"

# Create or open the file in append mode
with open(output_file, "a") as file:
    # Append the results to the file
    file.write(f'Results for bin_factor = {i}\n\n')
    file.write(f'Average Mean Squared Error (MSE) across all folds: {average_mse}\n')
    file.write(f'Average Accuracy across all folds: {average_accuracy}\n')
    file.write('_____________\n\n')
    
    # Make sure to close the file when you're done
    file.close()

