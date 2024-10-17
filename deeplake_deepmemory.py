import openai
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


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


def retrieve_context_from_deeplake(vector_store_db, user_question, deep_memory):
    # deep memory inside the vectore store ==> deep_memory=True
    answer = vector_store_db.search(
        embedding_data=user_question,
        embedding_function=embedding_function,
        deep_memory=deep_memory,
        return_view=False,
        k=4,
    )
    return answer
