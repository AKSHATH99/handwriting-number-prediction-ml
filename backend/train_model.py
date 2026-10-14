import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

def train_and_save_model():
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    print("Building model...")
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    model_path = Path(__file__).resolve().parent / "mnist_model.keras"
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
