import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import Preprocessing
import time

start_time = time.time()

path_BH = './data/BH/'
path_NS = './data/NS/'
preprocessor = Preprocessing() 
BH_powerspectra = preprocessor.collect_all_powerspectra(path_BH, bin_factor=10, BH=True)
NS_powerspectra = preprocessor.collect_all_powerspectra(path_NS, bin_factor=10, BH=False)

data_array=np.vstack([BH_powerspectra,NS_powerspectra])
data=pd.DataFrame(data_array,columns=['freq','power','error','BH?'])

post_processing_time = time.time()

X = data[['freq', 'power']]
y = data['BH?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200,min_samples_leaf=20,min_samples_split=50)

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
print("Tiempo de procesamiento de archivos:",post_processing_time - start_time, "segundos")
print("Tiempo de entrenamiento:", time.time() - start_time, "segundos")
print()
