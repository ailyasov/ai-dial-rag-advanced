import requests

from task._constants import API_KEY

DIAL_EMBEDDINGS = "https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings"


# TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)


class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str) -> None:
        self.deployment_name = deployment_name
        self.api_key = api_key

    def get_embeddings(self, input_list: list) -> dict:
        url = DIAL_EMBEDDINGS.format(model=self.deployment_name)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "input": input_list,
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json().get("data", [])
        return {item["index"]: item["embedding"] for item in data}


def main():
    # Example usage
    deployment_name = "text-embedding-3-small-1"
    api_key = API_KEY
    client = DialEmbeddingsClient(deployment_name, api_key)
    input_texts = ["Hello, my dog is cute", "What is the capital of France?"]
    embeddings = client.get_embeddings(input_texts)
    print(embeddings)


if __name__ == "__main__":
    main()

# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
