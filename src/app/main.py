import asyncio

from src.bootstrap import build_container


async def main() -> None:
    container = build_container()

    if not await container.elasticsearch.ping():
        raise RuntimeError("Elasticsearch is not running")

    print("Elasticsearch is running")

    texts = ["This is the test to check if embedding works", "This is the second test to check if embedding works"]

    embedding = container.embedding_model
    embeddings = await embedding.embed(texts)
    print(embeddings)

def run() -> None:
    asyncio.run(main())
