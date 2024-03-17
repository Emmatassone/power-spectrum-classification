import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

path_BH = os.path.join('data_test', 'BH')
path_NS = os.path.join('data_test', 'NS')
preprocessor = Preprocessing() 

print("I'm here: Step 2!\n")

bin_factor=100
BH_powerspectra = preprocessor.array_collect(path_BH, bin_factor=bin_factor, BH=True)
NS_powerspectra = preprocessor.array_collect(path_NS, bin_factor=bin_factor, BH=False)
powerspectra=np.vstack((BH_powerspectra,NS_powerspectra))

print("I'm here: Step 3!\n")

X_test = powerspectra[:,0:3]  # Original: [:,0:2]
y_test = powerspectra[:,3]

print("I'm here: Step 4!\n")

np.random.shuffle(powerspectra)

X_train = powerspectra[:,0:3]  # Original: [:,0:2]
y_train = powerspectra[:,3]

ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

post_processing_time = time.time()

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200,min_samples_leaf=20,min_samples_split=50)

print("I'm here: Step 5!\n")

# Train the model
rf_classifier.fit(X_train, y_train)

print("I'm here: Step 6!\n")

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

print("I'm here: Step 7!\n")

# Calculate r^2 and mean squared error
mse = mean_squared_error(y_test, y_pred)

print("I'm here: Step 8!\n")

# Print the results
print(f'Mean Squared Error (MSE): {mse}')

print('\n\n' + str(confusion_matrix(y_test, y_pred)) + '\n\n'
      + str(accuracy_score(y_test, y_pred)) + '\n'
      + '_____________' + '\n\n'
)

i = 100
print()
print("Tiempo de procesamiento de archivos:", post_processing_time - start_time, "segundos")
print("Tiempo de entrenamiento con bineado", i, ":", time.time() - post_processing_time, "segundos")
#print("Tiempo de entrenamiento:", time.time() - post_processing_time, "segundos")

print()

model_info = {'model': rf_classifier,
              'accuracy': str(accuracy_score(y_test, y_pred)),
              'rebin_factor':str(i),
              'reading_time': post_processing_time - start_time,
              'training_time':  time.time() - post_processing_time,
              'confusion_matrix':str(confusion_matrix(y_test, y_pred))}

#joblib.dump(model_info, 'rf_classifier'+str(bin_factor)+'.pkl')
joblib.dump(model_info, 'rf_classifier'+str(i)+'.pkl')

# G added these lines
# Define a file path where you want to save the results
output_file = "results_PS_RF.txt"

# Create or open the file in append mode
with open(output_file, "a") as file:
    # Append the results to the file
#    file.write(f'Results for bin_factor = {bin_factor}\n\n')
    file.write(f'Results for bin_factor = {i}\n\n')
    file.write(f'Mean Squared Error (MSE): {mse}\n\n')
        
    # You can also format the confusion matrix and accuracy score as needed
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    file.write(f'Confusion Matrix:\n{cm}\n\n')
    file.write(f'Accuracy Score: {accuracy}\n')
    file.write('_____________\n\n')

    # Make sure to close the file when you're done
    file.close()

