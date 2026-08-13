import json, re
import numpy as np
import tensorflow as tf
from pathlib import Path

DATASET_PATH = Path("../dataset/intents.json")
MODEL_PATH = Path("../model/cnn_chatbot.keras")
VOCAB_PATH = Path("../model/cnn_vocab.json")
TFLITE_PATH = Path("../model/cnn_chatbot_int8.tflite")

MAX_LENGTH = 12

# ==========================================
# 1. Load Dataset & Build Vocabulary
# ==========================================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

vocab_set = set()
for intent in data["intents"]:
    for example in intent["examples"]:
        words = re.findall(r"[a-z0-9]+", example.lower())
        vocab_set.update(words)

vocabulary = sorted(vocab_set)
word_to_id = {word: i for i, word in enumerate(vocabulary)}
vocab_size = len(vocabulary)

labels = [intent["name"] for intent in data["intents"]]
label_to_id = {label: i for i, label in enumerate(labels)}

# ==========================================
# 2. Create One-Hot Sequences
# ==========================================
def encode_sequence(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Create a 2D array of 0s
    seq = np.zeros((MAX_LENGTH, vocab_size), dtype=np.float32)
    for i, word in enumerate(words[:MAX_LENGTH]):
        if word in word_to_id:
            seq[i, word_to_id[word]] = 1.0 # Mark word presence
    return seq

X, y = [], []
for intent in data["intents"]:
    for example in intent["examples"]:
        X.append(encode_sequence(example))
        y.append(label_to_id[intent["name"]])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# ==========================================
# 3. Build 1D-CNN Model (No Embedding Layer)
# ==========================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH, vocab_size)),
    tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(labels), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=150, batch_size=4, verbose=1)

# ==========================================
# 4. Save Vocabulary
# ==========================================
with open(VOCAB_PATH, "w") as f:
    json.dump({"vocabulary": vocabulary, "labels": labels, "max_length": MAX_LENGTH}, f, indent=2)

# ==========================================
# 5. INT8 Quantization
# ==========================================
def representative_dataset():
    for sample in X:
        yield [np.expand_dims(sample, axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Keep I/O as FLOAT32 to keep the Arduino C++ code incredibly simple
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
TFLITE_PATH.write_bytes(tflite_model)

print(f"\nCreated INT8 Model: {len(tflite_model)} bytes")

# ==========================================
# 6. Generate C Array
# ==========================================
with open("../model/model_data.cc", "w") as f:
    f.write('#include <stdint.h>\n\nconst unsigned char model_data[] = {\n')
    f.write(','.join(f'0x{b:02x}' for b in tflite_model))
    f.write('\n};\n\nconst unsigned int model_data_len = ' + str(len(tflite_model)) + ';\n')
print("Generated model_data.cc for ESP32")