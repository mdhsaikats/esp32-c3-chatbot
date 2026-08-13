import json
import re
import numpy as np
import tensorflow as tf
from pathlib import Path


# ==========================================
# Paths
# ==========================================

DATASET_PATH = Path("../dataset/intents.json")
MODEL_PATH = Path("../model/bow_chatbot.keras")
VOCAB_PATH = Path("../model/bow_vocab.json")
TFLITE_PATH = Path("../model/bow_chatbot_int8.tflite")


# ==========================================
# Configuration
# ==========================================

MAX_WORDS = 100


# ==========================================
# Tokenizer
# ==========================================

def tokenize(text):
    return re.findall(
        r"[a-z0-9]+",
        text.lower()
    )


# ==========================================
# Load dataset
# ==========================================

print("Loading dataset...")

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


# ==========================================
# Build vocabulary
# ==========================================

word_counts = {}

for intent in data["intents"]:

    for example in intent["examples"]:

        words = tokenize(example)

        for word in words:

            word_counts[word] = (
                word_counts.get(word, 0) + 1
            )


# Sort by frequency

vocabulary = sorted(
    word_counts,
    key=word_counts.get,
    reverse=True
)[:MAX_WORDS]


word_to_id = {
    word: i
    for i, word in enumerate(vocabulary)
}


print()
print("Vocabulary size:", len(vocabulary))


# ==========================================
# Labels
# ==========================================

labels = [
    intent["name"]
    for intent in data["intents"]
]

label_to_id = {
    label: i
    for i, label in enumerate(labels)
}


print("Number of intents:", len(labels))


# ==========================================
# Convert sentence → BOW vector
# ==========================================

def encode_text(text):

    vector = np.zeros(
        len(vocabulary),
        dtype=np.float32
    )

    for word in tokenize(text):

        if word in word_to_id:

            vector[
                word_to_id[word]
            ] = 1.0

    return vector


# ==========================================
# Create training data
# ==========================================

X = []
y = []


for intent in data["intents"]:

    intent_id = label_to_id[
        intent["name"]
    ]

    for example in intent["examples"]:

        X.append(
            encode_text(example)
        )

        y.append(
            intent_id
        )


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)


# ==========================================
# Shuffle
# ==========================================

np.random.seed(42)

indices = np.arange(
    len(X)
)

np.random.shuffle(
    indices
)

X = X[indices]
y = y[indices]


print()
print("Training samples:", len(X))
print("Input shape:", X.shape)


# ==========================================
# Build tiny model
# ==========================================

model = tf.keras.Sequential([

    tf.keras.Input(
        shape=(len(vocabulary),)
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        len(labels),
        activation="softmax"
    )
])


# ==========================================
# Compile
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.003
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


print()
print("================================")
print("MODEL")
print("================================")

model.summary()


# ==========================================
# Train
# ==========================================

print()
print("Training...")


model.fit(

    X,
    y,

    epochs=150,

    batch_size=8,

    validation_split=0.2,

    verbose=1,

    callbacks=[

        tf.keras.callbacks.EarlyStopping(

            monitor="val_accuracy",

            patience=20,

            restore_best_weights=True
        )

    ]
)


# ==========================================
# Evaluate
# ==========================================

loss, accuracy = model.evaluate(
    X,
    y,
    verbose=0
)


print()
print("================================")
print("TRAINING COMPLETE")
print("================================")

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ==========================================
# Save Keras model
# ==========================================

model.save(
    MODEL_PATH
)

print(
    "Saved:",
    MODEL_PATH
)


# ==========================================
# Save vocabulary
# ==========================================

vocab_data = {

    "vocabulary": vocabulary,

    "word_to_id": word_to_id,

    "labels": labels
}


with open(
    VOCAB_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        vocab_data,
        f,
        indent=2
    )


print(
    "Saved:",
    VOCAB_PATH
)


# ==========================================
# Representative dataset
# ==========================================

def representative_dataset():

    # Use actual training examples

    for sample in X:

        yield [
            sample.reshape(
                1,
                -1
            ).astype(
                np.float32
            )
        ]


# ==========================================
# INT8 conversion
# ==========================================

print()
print("Converting to INT8...")


converter = (
    tf.lite.TFLiteConverter
    .from_keras_model(model)
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = (
    representative_dataset
)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = (
    tf.int8
)

converter.inference_output_type = (
    tf.int8
)


tflite_model = converter.convert()


# ==========================================
# Save INT8 model
# ==========================================

TFLITE_PATH.write_bytes(
    tflite_model
)


print()
print("================================")
print("INT8 CONVERSION COMPLETE")
print("================================")

print(
    "File:",
    TFLITE_PATH
)

print(
    "Size:",
    len(tflite_model),
    "bytes"
)

print(
    "Size:",
    round(
        len(tflite_model) / 1024,
        2
    ),
    "KB"
)