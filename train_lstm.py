import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Parameters
SEQUENCE_LENGTH = 25  # Updated to 25 frames
DATA_DIR = "seq_data_v2"  # New directory for 252-feature data
FEATURES_PER_FRAME = 252  # 2 hands x 42 landmarks x 3 (x, y, z)
BATCH_SIZE = 32  # Standard batch size for training
EPOCHS = 100  # Maximum epochs with early stopping
TEST_SIZE = 0.2  # 20% data for validation
RANDOM_STATE = 42  # For reproducibility

# Load dataset
print(f"[🔍] Loading data from {DATA_DIR}...")
X, y = [], []
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f"[⚠️] Skipping non-directory: {folder}")
        continue
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".npy"):
            seq = np.load(file_path)
            # Validate shape to ensure consistency with 252 features
            if seq.shape[1] == FEATURES_PER_FRAME and len(seq) == SEQUENCE_LENGTH:
                X.append(seq)
                y.append(folder)
            else:
                print(f"[⚠️] Skipping {file} due to shape mismatch: {seq.shape}")

X = np.array(X)
y = np.array(y)
print(f"[✅] Data loaded. Total sequences: {len(X)}")

# Debug: Verify class distribution
unique_signs = np.unique(y)
print(f"[DEBUG] Loaded {len(unique_signs)} unique signs: {unique_signs}")
print(f"[DEBUG] Samples per class: {Counter(y)}")

# Encode gesture labels
print("[🔧] Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
NUM_CLASSES = len(np.unique(y_encoded))
print(f"[✅] Encoded {NUM_CLASSES} classes.")

# Split into training and validation sets
print("[🔪] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
)
print(f"[✅] Split: Train {len(X_train)}, Test {len(X_test)}")

# Build the LSTM model
print("[🏗️] Building LSTM model...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME),
         name="lstm_layer_1"),
    Dropout(0.2, name="dropout_1"),  # Added for regularization
    LSTM(64, name="lstm_layer_2"),
    Dropout(0.2, name="dropout_2"),  # Added for regularization
    Dense(min(64, NUM_CLASSES * 4), activation='relu', name="dense_hidden"),
    Dense(NUM_CLASSES, activation='softmax', name="output_layer")
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"[✅] Model compiled with input shape: {(SEQUENCE_LENGTH, FEATURES_PER_FRAME)}")

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
print("[🛡️] Early stopping configured.")

# Train the model
print("[🚀] Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    batch_size=BATCH_SIZE,
    verbose=1
)
print("[✅] Training completed.")

# Save model and label encoder
print("[💾] Saving model and label encoder...")
model.save("lstm_model.h5")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"[✅] Model and label encoder saved. Trained on {NUM_CLASSES} classes.")