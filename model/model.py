import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc


# ✅ Data Preprocessing
def load_and_preprocess_data(train_filename, test_filename):
    """
    Loads, preprocesses, and balances training and test data.
    
    Parameters:
    train_filename (str): Path to training dataset (CSV file).
    test_filename (str): Path to test dataset (CSV file).
    
    Returns:
    X_train, X_val, y_train, y_val, X_test, label_encoder: Processed datasets and label encoder.
    """
    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)

    # Keep only numeric features
    numeric_features = train_df.select_dtypes(include=[np.number]).columns
    X = train_df[numeric_features].values  
    y = train_df.iloc[:, -1].values  # Target variable

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess test set
    X_test = test_df[numeric_features].values
    X_test = scaler.transform(X_test)

    return X_train, X_val, y_train, y_val, X_test, label_encoder


# ✅ Model Definition
def build_hybrid_model(input_shape, num_classes):
    """
    Constructs a CNN + LSTM hybrid deep learning model.
    
    Parameters:
    input_shape (int): Number of features (input dimension).
    num_classes (int): Number of output classes.
    
    Returns:
    model: Compiled Keras model.
    """
    model = keras.Sequential([
        keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

        # CNN Layers
        keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
        keras.layers.Conv1D(128, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),

        # LSTM Layer
        keras.layers.Reshape((1, -1)),  
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.LSTM(64),

        # Fully Connected Layers
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ✅ Model Training
def train_model(model, X_train, y_train, X_val, y_val):
    """
    Trains the CNN + LSTM model with early stopping and learning rate reduction.
    
    Parameters:
    model: Compiled Keras model.
    X_train, y_train: Training data.
    X_val, y_val: Validation data.
    
    Returns:
    history: Training history for visualization.
    """
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)

    history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr])
    return history


# ✅ Model Evaluation
def evaluate_model(model, X_val, y_val, label_encoder):
    """
    Generates a classification report and AUC-ROC curve.
    
    Parameters:
    model: Trained Keras model.
    X_val, y_val: Validation dataset.
    label_encoder: Label encoder object.
    """
    y_val_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_probs, axis=1)
    print(classification_report(y_val, y_val_pred, zero_division=1))

    # AUC-ROC Curve
    y_val_binary = keras.utils.to_categorical(y_val)
    for i in range(y_val_binary.shape[1]):
        fpr, tpr, _ = roc_curve(y_val_binary[:, i], y_val_probs[:, i])
        plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC = {auc(fpr, tpr):.2f})")
    
    plt.legend()
    plt.savefig("auc_roc_curve.png")
    plt.show()
# ✅ Plot and Save Validation Loss
def plot_validation_loss(history):
    """
    Plots validation loss over epochs and saves the image.
    """
    plt.figure(figsize=(8,5))
    plt.plot(history.history['val_loss'], label="Validation Loss", color="red")
    plt.plot(history.history['loss'], label="Training Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("validation_loss.png")
    plt.show()

# ✅ Save Predictions to CSV
def save_predictions(model, X_test, label_encoder, output_filename="test_predictions.csv"):
    """
    Saves the test dataset predictions in a CSV file with ID and detected clade.
    """
    y_test_probs = model.predict(X_test)
    y_test_pred = np.argmax(y_test_probs, axis=1)
    predicted_clades = label_encoder.inverse_transform(y_test_pred)
    
    # Creating dataframe with IDs and predicted clades
    results_df = pd.DataFrame({
        "ID": test_df["ID"],  # Assuming test_df has an index column
        "Clade": predicted_clades
    })
    
    results_df.to_csv(output_filename, index=False)
    print(f"✅ Predictions saved to {output_filename}")