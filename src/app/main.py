import asyncio

from src.bootstrap import build_container


async def main() -> None:
    container = build_container()

    if not await container.elasticsearch.ping():
        raise RuntimeError("Elasticsearch is not running")

    print("Elasticsearch is running")


def run() -> None:
    asyncio.run(main())
