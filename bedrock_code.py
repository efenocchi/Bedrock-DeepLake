# Use the Converse API to send a text message to Claude 3 Sonnet.

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
from utils import (
    get_image_message_structure,
    get_text_message_structure,
    image_to_bytes,
)

load_dotenv()
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"


def get_bedrock_answer_with_images(question, image):
    # Start a conversation with the user message.
    messages = get_image_message_structure(image, question)

    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages,
            inferenceConfig={"maxTokens": 2000, "temperature": 0},
            additionalModelRequestFields={"top_k": 250},
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)
        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)


def get_bedrock_answer_with_text(question, chunks):
    # Start a conversation with the user message.
    messages = get_text_message_structure(chunks, question)
    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages,
            inferenceConfig={"maxTokens": 2000, "temperature": 0},
            additionalModelRequestFields={"top_k": 250},
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)
        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)


if __name__ == "__main__":
    byte_image = image_to_bytes("image.png")
    question = "describe the image"
    get_bedrock_answer_with_images(question, byte_image)
