from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float64, String

card = Entity(name="card", join_keys=["card1"])

batch_source = FileSource(
    name="card_window_batch",
    path="data/card_window.parquet",
    timestamp_field="event_timestamp",
)

card_window_push = PushSource(name="card_window_push", batch_source=batch_source)

card_window = FeatureView(
    name="card_window",
    entities=[card],
    ttl=timedelta(days=1),
    online=True,
    source=card_window_push,
    schema=[
        Field(name="card1", dtype=String),
        Field(name="win_count", dtype=Float64),
        Field(name="win_sum", dtype=Float64),
        Field(name="win_velocity", dtype=Float64),
        Field(name="win_delta", dtype=Float64),
    ],
)
