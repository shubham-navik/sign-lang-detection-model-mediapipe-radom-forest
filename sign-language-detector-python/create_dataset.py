import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Loop through directories in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Check if the entry is a directory
    if os.path.isdir(os.path.join(DATA_DIR, dir_)):
        # Loop through files in the directory
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

# Pad sequences to the maximum length
max_len = max(len(seq) for seq in data)
data_padded = np.array([np.pad(seq, (0, max_len - len(seq)), mode='constant') for seq in data])

# Convert labels to integers
labels = np.array(labels, dtype=int)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data_padded, 'labels': labels}, f)
