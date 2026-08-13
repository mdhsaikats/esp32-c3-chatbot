import json
import random
import numpy as np
import tensorflow as tf
from pathlib import Path


MODEL_PATH = Path("../model/intent_model.keras")
VOCAB_PATH = Path("../model/intent_vocab.json")
DATASET_PATH = Path("../dataset/intents.json")


# ==========================================
# Load
# ==========================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:

    vocab_data = json.load(f)


with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


char_to_id = vocab_data["char_to_id"]
labels = vocab_data["labels"]
MAX_LENGTH = vocab_data["max_length"]


# ==========================================
# Encode
# ==========================================

def encode(text):

    text = text.lower()

    result = np.zeros(
        MAX_LENGTH,
        dtype=np.int32
    )

    for i, char in enumerate(
        text[:MAX_LENGTH]
    ):

        result[i] = char_to_id.get(
            char,
            0
        )

    return result


# ==========================================
# Predict
# ==========================================

def predict_intent(text):

    encoded = encode(text)

    encoded = np.expand_dims(
        encoded,
        axis=0
    )

    probabilities = model.predict(
        encoded,
        verbose=0
    )[0]

    index = np.argmax(
        probabilities
    )

    confidence = probabilities[index]

    return labels[index], confidence


# ==========================================
# Responses
# ==========================================

responses = {

    intent["name"]:
        intent["responses"]

    for intent in dataset["intents"]
}


# ==========================================
# Chat
# ==========================================

print()
print("================================")
print("       C3 AI Chatbot")
print("================================")
print("Type 'exit' to quit.")
print()


while True:

    text = input("You: ").strip()

    if text.lower() == "exit":
        break

    if not text:
        continue

    intent, confidence = predict_intent(
        text
    )

    response = random.choice(
        responses[intent]
    )

    print(
        f"Intent: {intent}"
    )

    print(
        f"Confidence: {confidence:.2f}"
    )

    print(
        f"Bot: {response}"
    )

    print()