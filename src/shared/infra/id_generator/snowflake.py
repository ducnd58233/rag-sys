from __future__ import annotations

from snowflake import SnowflakeGenerator


class SnowflakeIdGenerator:
    def __init__(self, instance_id: int) -> None:
        if not 0 <= instance_id < 1024:
            raise ValueError("Instance ID must be between 0 and 1023")

        self._generator = SnowflakeGenerator(instance_id)

    def next_id(self) -> int:
        identifier = next(self._generator)

        if identifier is None or identifier <= 0:
            raise RuntimeError("Snowflake generator returned an invalid ID")

        return identifier
