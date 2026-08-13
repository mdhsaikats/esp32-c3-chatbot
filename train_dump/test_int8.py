import json
import numpy as np
import tensorflow as tf
from pathlib import Path


# ==========================================
# Paths
# ==========================================

MODEL_PATH = Path("../model/intent_model_int8.tflite")
VOCAB_PATH = Path("../model/intent_vocab.json")


# ==========================================
# Load vocabulary
# ==========================================

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:

    vocab_data = json.load(f)


char_to_id = vocab_data["char_to_id"]
labels = vocab_data["labels"]
MAX_LENGTH = vocab_data["max_length"]


# ==========================================
# Load INT8 TFLite model
# ==========================================

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()


input_details = interpreter.get_input_details()[0]

output_details = interpreter.get_output_details()[0]


# ==========================================
# Quantization parameters
# ==========================================

input_scale, input_zero = (
    input_details["quantization"]
)

output_scale, output_zero = (
    output_details["quantization"]
)


print()
print("================================")
print("       INT8 MODEL INFO")
print("================================")

print(
    "Input shape:",
    input_details["shape"]
)

print(
    "Input type:",
    input_details["dtype"]
)

print(
    "Input scale:",
    input_scale
)

print(
    "Input zero point:",
    input_zero
)

print()

print(
    "Output shape:",
    output_details["shape"]
)

print(
    "Output type:",
    output_details["dtype"]
)

print(
    "Output scale:",
    output_scale
)

print(
    "Output zero point:",
    output_zero
)


# ==========================================
# Encode text
# ==========================================

def encode(text):

    text = text.lower()

    values = np.zeros(
        MAX_LENGTH,
        dtype=np.float32
    )

    for i, char in enumerate(
        text[:MAX_LENGTH]
    ):

        values[i] = char_to_id.get(
            char,
            0
        )

    return values


# ==========================================
# Predict with INT8 model
# ==========================================

def predict(text):

    # --------------------------------------
    # Encode text
    # --------------------------------------

    float_input = encode(text)


    # --------------------------------------
    # FLOAT32 → INT8
    # --------------------------------------

    int8_input = np.round(
        float_input / input_scale
        + input_zero
    )

    int8_input = np.clip(
        int8_input,
        -128,
        127
    ).astype(
        np.int8
    )


    # Add batch dimension

    int8_input = np.expand_dims(
        int8_input,
        axis=0
    )


    # --------------------------------------
    # Run model
    # --------------------------------------

    interpreter.set_tensor(
        input_details["index"],
        int8_input
    )

    interpreter.invoke()


    # --------------------------------------
    # Get INT8 output
    # --------------------------------------

    raw_output = interpreter.get_tensor(
        output_details["index"]
    )


    # --------------------------------------
    # INT8 → FLOAT32
    # --------------------------------------

    probabilities = (

        raw_output.astype(
            np.float32
        )

        - output_zero

    ) * output_scale


    # --------------------------------------
    # Find best intent
    # --------------------------------------

    index = np.argmax(
        probabilities[0]
    )


    confidence = probabilities[
        0,
        index
    ]


    return (
        labels[index],
        float(confidence)
    )


# ==========================================
# Test sentences
# ==========================================

tests = [

    "hello",

    "hi",

    "hey",

    "who are you",

    "what is your name",

    "tell me who you are",

    "do you need wifi",

    "do you need internet",

    "are you online",

    "what is esp32",

    "what is esp32 c3",

    "what is ai",

    "what does ai mean",

    "thank you",

    "thanks",

    "bye",

    "goodbye"

]


# ==========================================
# Run tests
# ==========================================

print()
print("================================")
print("       INT8 MODEL TEST")
print("================================")


correct = 0


for text in tests:

    intent, confidence = predict(
        text
    )


    print()

    print(
        "Input:",
        text
    )

    print(
        "Intent:",
        intent
    )

    print(
        "Confidence:",
        round(
            confidence,
            3
        )
    )


print()
print("================================")
print("          TEST COMPLETE")
print("================================")
print()