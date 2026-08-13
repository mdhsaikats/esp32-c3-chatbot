import numpy as np
import tensorflow as tf
from pathlib import Path


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = Path("../model/chatbot.keras")
VOCAB_PATH = Path("../model/vocab.txt")

SEQ_LENGTH = 64

MAX_RESPONSE_LENGTH = 100


# ==========================================
# Load model
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded!")


# ==========================================
# Load vocabulary
# ==========================================

vocab = VOCAB_PATH.read_text(
    encoding="utf-8"
)

char_to_id = {
    char: i
    for i, char in enumerate(vocab)
}

id_to_char = {
    i: char
    for i, char in enumerate(vocab)
}


# ==========================================
# Generate response
# ==========================================

def generate_response(user_text):

    prompt = (
        "<user> "
        + user_text
        + " <bot> "
    )

    generated = prompt

    for _ in range(
        MAX_RESPONSE_LENGTH
    ):

        context = generated[
            -SEQ_LENGTH:
        ]

        encoded = [
            char_to_id.get(
                c,
                0
            )
            for c in context
        ]

        # Padding

        if len(encoded) < SEQ_LENGTH:

            encoded = (

                [0] *
                (
                    SEQ_LENGTH
                    - len(encoded)
                )

                + encoded
            )


        input_data = np.array(
            [encoded],
            dtype=np.int32
        )


        # Predict

        logits = model.predict(
            input_data,
            verbose=0
        )[0]


        # Softmax

        probabilities = (
            tf.nn.softmax(
                logits
            ).numpy()
        )


        # Select most likely character

        next_id = np.argmax(
            probabilities
        )

        next_char = id_to_char[
            next_id
        ]


        generated += next_char


        # Stop when model starts
        # another conversation

        if (
            "<user>"
            in generated
            or "\n\n"
            in generated
        ):
            break


    # Extract bot response

    if "<bot>" in generated:

        response = generated.split(
            "<bot>",
            1
        )[1]

    else:

        response = generated


    # Remove conversation markers

    response = response.replace(
        "<user>",
        ""
    )

    response = response.replace(
        "<bot>",
        ""
    )

    response = response.strip()


    return response


# ==========================================
# Chat
# ==========================================

print()
print("================================")
print("       C3 Tiny Chatbot")
print("================================")
print("Type 'exit' to quit.")
print()


while True:

    user_input = input(
        "You: "
    ).strip()


    if user_input.lower() == "exit":

        print("Goodbye!")

        break


    if not user_input:

        continue


    response = generate_response(
        user_input
    )


    print(
        "Bot:",
        response
    )

    print()