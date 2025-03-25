import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# Load the Diabetes dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Normalize features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['target']))

# Convert target to binary classification (above or below median value)
y = (df['target'] > df['target'].median()).astype(int)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a Sequential model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer with 16 neurons
    Dropout(0.2),  # Dropout to prevent overfitting
    Dense(8, activation='relu'),  # Second hidden layer with 8 neurons
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with 50 epochs and a batch size of 16
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")