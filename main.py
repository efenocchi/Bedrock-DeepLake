import base64
from io import BytesIO
import json
import requests
import os
from PIL import Image
from dotenv import load_dotenv

from bedrock_code import get_bedrock_answer_with_images, get_bedrock_answer_with_text
from deeplake_deepmemory import retrieve_context_from_deeplake
from deeplake.core.vectorstore import VectorStore
import deeplake

load_dotenv()
os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")


def retrieve_data(
    queries: list, org_id: str, dataset_name: str, k=4, number_of_images=3
):
    url = f"https://beta.activeloop.dev/api/query/colpali/{org_id}/{dataset_name}"

    data = {
        "queries": queries,
        "k": k,
        "number_of_images": number_of_images,
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('TOKEN')}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def save_images(value_returned: dict):
    for idx_question, img_list in enumerate(value_returned["images"]):
        for idx_img, img in enumerate(img_list):
            image_data = base64.b64decode(img)
            image = Image.open(BytesIO(image_data))
            image.save(f"question_{idx_question}_image_{idx_img}.jpg")


def send_request(query: list, org_id: str, dataset_name: str):

    value_returned = retrieve_data(query, org_id, dataset_name)
    save_images(value_returned)

    for img_list in value_returned["images"]:
        for img in img_list:
            byte_image = base64.b64decode(img)
            answer = get_bedrock_answer_with_images(query, byte_image)
            print("the answer is: ", answer)
            break


if __name__ == "__main__":
    org_id = "emanuelebeta"
    dataset_name = "ingestion_ml_test2_colpali"
    questions = "describe the gaussian distribution curve"
    send_request(questions, org_id, dataset_name)

    # deeplake
    question = "How does the choice of T value affect the calculation of daily CFRs in the context of disease progression?"

    vector_store = VectorStore(
        "hub://activeloop/biomed_deep_memory_project_24", read_only=True
    )

    deep_memory_chunks = retrieve_context_from_deeplake(
        vector_store, question, deep_memory=True
    )
    no_deep_memory_chunks = retrieve_context_from_deeplake(
        vector_store, question, deep_memory=False
    )

    final_answer_deep_memory = get_bedrock_answer_with_text(
        question, deep_memory_chunks
    )
    final_answer_no_deep_memory = get_bedrock_answer_with_text(
        question, no_deep_memory_chunks
    )
    print(f"final answer deep memory: {final_answer_deep_memory}")
    print(f"final answer no deep memory: {final_answer_no_deep_memory}")
