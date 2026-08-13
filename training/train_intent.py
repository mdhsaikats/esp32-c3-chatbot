import json
import numpy as np
import tensorflow as tf
from pathlib import Path


# ==========================================
# Configuration
# ==========================================

DATASET_PATH = Path("../dataset/intents.json")
MODEL_PATH = Path("../model/intent_model.keras")
VOCAB_PATH = Path("../model/intent_vocab.json")

MAX_LENGTH = 32

EMBEDDING_DIM = 8
HIDDEN_UNITS = 16

EPOCHS = 200


# ==========================================
# Load dataset
# ==========================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


intents = data["intents"]


# ==========================================
# Prepare examples
# ==========================================

texts = []
labels = []

label_names = []

for intent in intents:

    name = intent["name"]

    label_names.append(name)

    for example in intent["examples"]:

        texts.append(
            example.lower()
        )

        labels.append(name)


label_to_id = {
    name: i
    for i, name in enumerate(label_names)
}


y = np.array([
    label_to_id[label]
    for label in labels
])


# ==========================================
# Create character vocabulary
# ==========================================

all_text = "".join(texts)

vocab = sorted(
    set(all_text)
)


char_to_id = {
    char: i + 1
    for i, char in enumerate(vocab)
}


# 0 = padding / unknown

vocab_size = len(vocab) + 1


# ==========================================
# Encode text
# ==========================================

X = np.zeros(
    (
        len(texts),
        MAX_LENGTH
    ),
    dtype=np.int32
)


for i, text in enumerate(texts):

    for j, char in enumerate(
        text[:MAX_LENGTH]
    ):

        X[i, j] = char_to_id.get(
            char,
            0
        )


# ==========================================
# Shuffle dataset
# ==========================================

indices = np.arange(
    len(X)
)

np.random.shuffle(
    indices
)

X = X[indices]
y = y[indices]


# ==========================================
# Build model
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(MAX_LENGTH,)
    ),

    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(
        HIDDEN_UNITS,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        len(label_names),
        activation="softmax"
    )
])


# ==========================================
# Compile
# ==========================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# Train
# ==========================================

model.summary()

print("\nTraining...\n")


model.fit(

    X,
    y,

    epochs=EPOCHS,

    batch_size=8,

    verbose=1,

    shuffle=True
)


# ==========================================
# Save model
# ==========================================

model.save(
    MODEL_PATH
)


# ==========================================
# Save vocabulary
# ==========================================

vocab_data = {

    "char_to_id": char_to_id,

    "labels": label_names,

    "max_length": MAX_LENGTH
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


print("\n================================")
print("Training complete!")
print("================================")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Vocabulary:",
    VOCAB_PATH
)