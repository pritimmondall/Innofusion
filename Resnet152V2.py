# Smart Rescue Drone - Training directly from an Internet Dataset
# Model: ResNet152V2
# This script is designed for Google Colab.
# Please ensure you have enabled a GPU Runtime before running.

#--------------------------------------------------------------------------
# STEP 0: SETUP AND IMPORTS
#--------------------------------------------------------------------------
print("STEP 0: Setting up the environment...")

# Import all required libraries
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from google.colab import drive
from tensorflow.keras.applications import ResNet152V2 # Changed from ResNet151V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Mount Google Drive
try:
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")

# Define a NEW project path for this ResNet151 run
project_path = '/content/drive/MyDrive/SmartRescueDrone_ResNet151_TFDS'
os.makedirs(project_path, exist_ok=True)
print(f"Project directory is set to a new folder: {project_path}")
print("-" * 60)


#--------------------------------------------------------------------------
# STEP 1: LOAD DATASET DIRECTLY FROM THE INTERNET
#--------------------------------------------------------------------------
print("STEP 1: Loading dataset directly from TensorFlow Datasets (TFDS)...")

# We will use the 'tf_flowers' dataset to demonstrate the technique.
# TFDS handles all downloading and caching automatically.
# Data is split: 70% for training, 15% for validation, 15% for testing.
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'tf_flowers',
    split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
    with_info=True,
    as_supervised=True,
)

# Get the class names from the dataset info
class_names = ds_info.features['label'].names
print(f"Dataset '{ds_info.name}' loaded with classes: {class_names}")
print(f"This multi-class problem will be adapted to a binary one for our use case.")
print("-" * 60)


#--------------------------------------------------------------------------
# STEP 2: PRE-PROCESS THE DATA FOR RESNET
#--------------------------------------------------------------------------
print("STEP 2: Pre-processing the data for binary classification...")

IMG_SIZE = 224 # ResNet models are typically trained on 224x224 images
BATCH_SIZE = 32

def preprocess_data(image, label):
    """Resizes, normalizes, and adapts the data for our binary problem."""
    # Resize the image for ResNet
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # Use the specific pre-processing function for ResNetV2
    image = resnet_preprocess(image)

    # --- ADAPTATION TO BINARY CLASSIFICATION ---
    # We'll designate 'roses' (class_index=2) as our "human" class (label 1).
    # All other flower types will be treated as the "no_human" class (label 0).
    new_label = tf.cond(tf.equal(label, 2), lambda: 1, lambda: 0)

    return image, tf.cast(new_label, tf.int32)

# Apply the preprocessing to each dataset and prepare batches
ds_train = ds_train.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

print("Data pre-processing and batching complete.")
print("-" * 60)


#--------------------------------------------------------------------------
# STEP 3: BUILD AND TRAIN THE RESNET152V2 MODEL
#--------------------------------------------------------------------------
print("STEP 3: Building and training the ResNet152V2 model...")

base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Model architecture built. Starting training...")
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=10
)

model_save_path = os.path.join(project_path, 'resnet151_tfds_detector.h5') # Keep the old name for consistency or change if preferred
model.save(model_save_path)
print(f"\nModel trained and saved to {model_save_path}")
print("-" * 60)


#--------------------------------------------------------------------------
# STEP 4: EVALUATE THE MODEL AND CALCULATE METRICS
#--------------------------------------------------------------------------
print("STEP 4: Evaluating model on the test set...")

# To evaluate, we iterate through the test set to get true labels and predictions.
y_true = []
y_pred_probs = []

for images, labels in ds_test:
    batch_probs = model.predict(images, verbose=0)
    y_pred_probs.extend(batch_probs.flatten())
    y_true.extend(labels.numpy())

# Convert lists to numpy arrays for calculations
y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred_binary = (y_pred_probs > 0.5).astype(int)

# Calculate all the required metrics
accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
mcc = matthews_corrcoef(y_true, y_pred_binary)
auc = roc_auc_score(y_true, y_pred_probs)

cm = confusion_matrix(y_true, y_pred_binary)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# --- Display Results ---
print("\n--- Model Performance Metrics (ResNet152V2 on TFDS Data) ---")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"MCC Score:   {mcc:.4f}")
print(f"AUC Score:   {auc:.4f}")
print("-------------------------------------------------------------")

print("\n--- Confusion Matrix ---")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Human (Other Flowers)', 'Human (Roses)'],
            yticklabels=['No Human (Other Flowers)', 'Human (Roses)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for ResNet152V2 (TFDS Data)')
plt.show()

print("\nTraining and evaluation using TensorFlow Datasets complete.")

'''
- Model Performance Metrics (ResNet152V2 on TFDS Data) ---
Accuracy:    0.9491
Precision:   0.8897
Recall:      0.9030
Specificity: 0.9639
F1 Score:    0.8963
MCC Score:   0.8626
AUC Score:   0.9846
'''
