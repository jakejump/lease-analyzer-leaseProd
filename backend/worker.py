from __future__ import annotations

from rq import Worker, Queue, Connection
from redis import Redis
import os


def main() -> None:
    url = os.getenv("REDIS_URL")
    conn = Redis.from_url(url) if url else Redis()
    with Connection(conn):
        worker = Worker([Queue("default")])
        worker.work()


if __name__ == "__main__":
    main()

from rq import Worker, Queue, Connection
from redis import Redis


def main() -> None:
    conn = Redis()
    with Connection(conn):
        w = Worker(["default"])
        w.work()


if __name__ == "__main__":
    main()


