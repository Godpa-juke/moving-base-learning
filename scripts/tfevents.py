#!/usr/bin/env python3
"""Minimal TFRecord/tfevents scalar reader.

Avoids a tensorboard dependency: parses just enough protobuf to pull
``Event.summary.value[].simple_value`` out of an rsl_rl run directory.
"""
from __future__ import annotations

import collections
import struct
from pathlib import Path


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield field, ("varint", value)
        elif wire == 1:
            yield field, ("f64", buf[i : i + 8])
            i += 8
        elif wire == 2:
            length, i = _varint(buf, i)
            yield field, ("bytes", buf[i : i + length])
            i += length
        elif wire == 5:
            yield field, ("f32", buf[i : i + 4])
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")


def read_scalars(path: str | Path) -> dict[str, list[tuple[int, float]]]:
    """Return ``{tag: [(step, value), ...]}`` for every scalar in an event file."""
    scalars: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    data = Path(path).read_bytes()
    i = 0
    while i + 12 <= len(data):
        length = struct.unpack("<Q", data[i : i + 8])[0]
        i += 12  # 8-byte length + 4-byte masked crc
        record = data[i : i + length]
        i += length + 4  # payload + 4-byte masked crc
        step = 0
        summary = None
        for field, (kind, value) in _fields(record):
            if field == 2 and kind == "varint":
                step = value
            elif field == 5 and kind == "bytes":
                summary = value
        if summary is None:
            continue
        for field, (kind, value) in _fields(summary):
            if field != 1 or kind != "bytes":
                continue
            tag = None
            simple = None
            for sub, (sub_kind, sub_value) in _fields(value):
                if sub == 1 and sub_kind == "bytes":
                    tag = sub_value.decode("utf8", "replace")
                elif sub == 2 and sub_kind == "f32":
                    simple = struct.unpack("<f", sub_value)[0]
            if tag is not None and simple is not None:
                scalars[tag].append((step, simple))
    return dict(scalars)


def read_run(run_dir: str | Path) -> dict[str, list[tuple[int, float]]]:
    """Merge every event file in a run directory, sorted by step."""
    merged: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    for event_file in sorted(Path(run_dir).glob("events.out.tfevents.*")):
        for tag, series in read_scalars(event_file).items():
            merged[tag].extend(series)
    return {tag: sorted(series) for tag, series in merged.items()}
