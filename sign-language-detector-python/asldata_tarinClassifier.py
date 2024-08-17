import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
data_dict = pickle.load(open('./asldata.pickle', 'rb'))

# Pad sequences to the same length
max_length = max(len(seq) for seq in data_dict['data'])
data_padded = [seq + [0] * (max_length - len(seq)) for seq in data_dict['data']]

# Convert to NumPy array
data = np.asarray(data_padded)
labels = np.asarray(data_dict['labels'])

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
with open('aslmodel.p', 'wb') as f:
    pickle.dump({'model': model}, f)
