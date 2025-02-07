import argparse

import pandas as pd
from model import load_and_preprocess_data, build_hybrid_model, train_model, evaluate_model, plot_validation_loss, save_predictions
import tensorflow as tf

def main():
    """ Main function to train and evaluate the model with user-defined file paths. """
    parser = argparse.ArgumentParser(description="Train CNN + LSTM Model")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train.csv file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test_features.csv file")
    args = parser.parse_args()

    # Load data
    X_train, X_val, y_train, y_val, X_test, label_encoder = load_and_preprocess_data(args.train_path, args.test_path)

    # Build and train model
    num_features = X_train.shape[1]
    num_classes = len(set(y_train))
    model = build_hybrid_model(input_shape=num_features, num_classes=num_classes)
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate model
    evaluate_model(model, X_val, y_val, label_encoder)
    plot_validation_loss(history)
    # Save trained model
    model.save("hybrid_cnn_lstm_model.h5")
    print("✅ Model training complete and saved as hybrid_cnn_lstm_model.h5!")
    save_predictions(model, X_test, label_encoder, pd.read_csv(args.test_path), output_filename="test_predictions.csv")

if __name__ == "__main__":
    main()
