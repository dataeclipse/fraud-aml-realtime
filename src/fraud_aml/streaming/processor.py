from __future__ import annotations

import argparse
import time
from typing import Any

from fraud_aml.config import Settings, get_settings
from fraud_aml.logging_config import configure_logging, get_logger
from fraud_aml.streaming.features import WindowState, update_window
from fraud_aml.streaming.schema import deserialize


def make_mapper(window_seconds: int) -> Any:
    def mapper(
        state: WindowState | None, event: dict[str, Any]
    ) -> tuple[WindowState, dict[str, Any]]:
        if state is None:
            state = WindowState()
        feats = update_window(
            state,
            int(event["TransactionDT"]),
            float(event["TransactionAmt"]),
            window_seconds=window_seconds,
        )
        emit = {
            "card1": str(event["card1"]),
            "TransactionID": event["TransactionID"],
            "publish_ts": event["publish_ts"],
            "processed_ts": time.time(),
            **feats,
        }
        return state, emit

    return mapper


def _parse(msg: Any) -> dict[str, Any]:
    return deserialize(msg.value)


def run_events(events: list[dict[str, Any]], *, window_seconds: int) -> list[dict[str, Any]]:
    states: dict[str, WindowState] = {}
    mapper = make_mapper(window_seconds)
    out: list[dict[str, Any]] = []
    for event in events:
        key = str(event["card1"])
        state = states.setdefault(key, WindowState())
        _, emit = mapper(state, event)
        out.append(emit)
    return out


def build_flow(settings: Settings, on_output: Any) -> Any:
    import bytewax.operators as op
    from bytewax.connectors.kafka import operators as kop
    from bytewax.dataflow import Dataflow

    flow = Dataflow("fraud-stream")
    stream = kop.input(
        "kafka", flow, brokers=[settings.kafka_bootstrap], topics=[settings.stream_topic]
    )
    parsed = op.map("parse", stream.oks, _parse)
    keyed = op.key_on("key", parsed, lambda event: str(event["card1"]))
    featured = op.stateful_map("window", keyed, make_mapper(settings.window_seconds))
    op.inspect("emit", featured, lambda _step, item: on_output(item[1]))
    return flow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bytewax stream processor: window features to Redis."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print features, do not write to Redis"
    )
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    configure_logging()
    log = get_logger("processor")
    settings = get_settings()

    lags: list[float] = []
    if args.dry_run:

        def on_output(item: dict[str, Any]) -> None:
            lags.append(item["processed_ts"] - item["publish_ts"])
    else:
        from fraud_aml.streaming.online_store import get_store, push_features

        store = get_store()

        def on_output(item: dict[str, Any]) -> None:
            push_features(store, [item])
            lags.append(item["processed_ts"] - item["publish_ts"])

    from bytewax.testing import run_main

    flow = build_flow(settings, on_output)
    run_main(flow)  # type: ignore[no-untyped-call]
    if lags:
        ordered = sorted(lags)
        p50 = ordered[len(ordered) // 2]
        log.info("processed", events=len(lags), lag_p50_s=round(p50, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
