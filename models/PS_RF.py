import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import Preprocessing
import time
import joblib

start_time = time.time()

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')

preprocessor = Preprocessing(path_BH, path_NS)
powerspectra = preprocessor.collect_all_NS_BH_data()

shuffled_PS = np.random.shuffle(powerspectra)
data=pd.DataFrame(powerspectra,columns=['freq','power','error','BH?'])

post_processing_time = time.time()

X = data[['freq', 'power']]
y = data['BH?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200,min_samples_leaf=20,min_samples_split=50,n_jobs=30)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate r^2 and mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the results
print(f'Mean Squared Error (MSE): {mse}')

print('\n\n' + str(confusion_matrix(y_test, y_pred)) + '\n\n'
                  + str(accuracy_score(y_test, y_pred)) + '\n'
                  + '_____________' + '\n\n'
                  )

print()
print("Tiempo de procesamiento de archivos:", post_processing_time - start_time, "segundos")
print("Tiempo de entrenamiento con bineado", i, ":", time.time() - post_processing_time, "segundos")
print()

model_info = {'model': rf_classifier,
          'accuracy': str(accuracy_score(y_test, y_pred)),
          'rebin_factor':str(i),
          'reading_time': post_processing_time - start_time,
          'training_time':  time.time() - post_processing_time,
          'confusion_matrix':str(confusion_matrix(y_test, y_pred))}

joblib.dump(model_info, 'rf_classifier'+str(bin_factor)+'.pkl')

# G added these lines
# Define a file path where you want to save the results
output_file = "results_PS_RF.txt"

# Create or open the file in append mode
with open(output_file, "a") as file:
    # Append the results to the file
    file.write(f'Results for bin_factor = {bin_factor}\n\n')
    file.write(f'Mean Squared Error (MSE): {mse}\n\n')

    # You can also format the confusion matrix and accuracy score as needed
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    file.write(f'Confusion Matrix:\n{cm}\n\n')
    file.write(f'Accuracy Score: {accuracy}\n')
    file.write('_____________\n\n')

    # Make sure to close the file when you're done
    file.close()

