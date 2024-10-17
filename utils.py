import openai
import os


def image_to_bytes(filename):
    with open(filename, "rb") as image_file:
        image_bytes = image_file.read()
    return image_bytes


def get_image_message_structure(image_bytes, question: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                {"text": "Given the image, answer the following question: " + question},
            ],
        }
    ]
    return messages


def get_text_message_structure(chunks, question: str):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"given these chunks: {chunks} \n\n answer the following question: {question}"
                }
            ],
        }
    ]
    return message


def embedding_function(texts, model="text-embedding-3-large"):

    if isinstance(texts, str):
        texts = [texts]

    try:
        texts = [t.replace("\n", " ") for t in texts]
    except:
        pass
    return [
        data.embedding
        for data in openai.embeddings.create(input=texts, model=model).data
    ]
