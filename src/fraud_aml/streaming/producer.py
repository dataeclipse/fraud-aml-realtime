from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from fraud_aml.config import get_settings
from fraud_aml.features.build import ID, TIME
from fraud_aml.logging_config import configure_logging, get_logger
from fraud_aml.streaming.features import AMOUNT, ENTITY
from fraud_aml.streaming.schema import serialize


class Sink(Protocol):
    def produce(self, topic: str, value: bytes) -> None: ...

    def flush(self) -> None: ...


def _kafka_sink(bootstrap: str) -> Any:
    from confluent_kafka import Producer

    return Producer({"bootstrap.servers": bootstrap})


def playback_delay(dt: int, base_dt: int, speedup: float) -> float:
    return max(0.0, (dt - base_dt) / speedup)


def replay(
    parquet_path: Path,
    *,
    topic: str,
    speedup: float,
    sink: Sink,
    limit: int | None = None,
    clock: Any = time.monotonic,
    sleep: Any = time.sleep,
    wall: Any = time.time,
) -> int:
    df = pd.read_parquet(parquet_path, columns=[ID, TIME, ENTITY, AMOUNT])
    df = df.sort_values(TIME, kind="stable")
    if limit is not None:
        df = df.head(limit)
    base_dt = int(df[TIME].iloc[0])
    start = clock()
    count = 0
    for row in df.itertuples(index=False):
        target = start + playback_delay(int(getattr(row, TIME)), base_dt, speedup)
        gap = target - clock()
        if gap > 0:
            sleep(gap)
        event = {
            "TransactionID": int(getattr(row, ID)),
            "TransactionDT": int(getattr(row, TIME)),
            "card1": str(getattr(row, ENTITY)),
            "TransactionAmt": float(getattr(row, AMOUNT)),
            "publish_ts": wall(),
        }
        sink.produce(topic, serialize(event))
        count += 1
    sink.flush()
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay IEEE-CIS transactions into Redpanda by time."
    )
    parser.add_argument("--speedup", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    configure_logging()
    log = get_logger("producer")
    settings = get_settings()
    speedup = args.speedup if args.speedup is not None else settings.speedup
    parquet = settings.data_dir / "interim" / "ieee_merged.parquet"
    sink = _kafka_sink(settings.kafka_bootstrap)
    count = replay(
        parquet, topic=settings.stream_topic, speedup=speedup, sink=sink, limit=args.limit
    )
    log.info("produced", count=count, topic=settings.stream_topic, speedup=speedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
