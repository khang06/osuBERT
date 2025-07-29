from __future__ import annotations

import dataclasses
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    SNAPPING = "snap"
    DISTANCE = "dist"
    NEW_COMBO = "new_combo"
    HITSOUND = "hitsound"
    VOLUME = "volume"
    CIRCLE = "circle"
    SPINNER = "spinner"
    SPINNER_END = "spinner_end"
    SLIDER_HEAD = "slider_head"
    BEZIER_ANCHOR = "bezier_anchor"
    PERFECT_ANCHOR = "perfect_anchor"
    CATMULL_ANCHOR = "catmull_anchor"
    RED_ANCHOR = "red_anchor"
    LAST_ANCHOR = "last_anchor"
    SLIDER_END = "slider_end"
    BEAT = "beat"
    MEASURE = "measure"
    TIMING_POINT = "timing_point"
    GAMEMODE = "gamemode"
    STYLE = "style"
    DIFFICULTY = "difficulty"
    MAPPER = "mapper"
    CS = "cs"
    YEAR = "year"
    HITSOUNDED = "hitsounded"
    SONG_LENGTH = "song_length"
    SONG_POSITION = "song_position"
    GLOBAL_SV = "global_sv"
    MANIA_KEYCOUNT = "keycount"
    HOLD_NOTE_RATIO = "hold_note_ratio"
    SCROLL_SPEED_RATIO = "scroll_speed_ratio"
    DESCRIPTOR = "descriptor"
    POS_X = "pos_x"
    POS_Y = "pos_y"
    POS = "pos"
    KIAI = "kiai"
    MANIA_COLUMN = "column"
    HOLD_NOTE = "hold_note"
    HOLD_NOTE_END = "hold_note_end"
    SCROLL_SPEED_CHANGE = "scroll_speed_change"
    SCROLL_SPEED = "scroll_speed"
    DRUMROLL = "drumroll"
    DRUMROLL_END = "drumroll_end"
    DENDEN = "denden"
    DENDEN_END = "denden_end"


class ContextType(Enum):
    NONE = "none"
    TIMING = "timing"
    NO_HS = "no_hs"
    GD = "gd"
    MAP = "map"
    KIAI = "kiai"
    SV = "sv"


@dataclasses.dataclass
class EventRange:
    type: EventType
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: EventType
    value: int = 0

    def __repr__(self) -> str:
        return f"{self.type.value}{self.value}"

    def __str__(self) -> str:
        return f"{self.type.value}{self.value}"

event_enum_to_idx: dict[EventType, bytes] = {}
event_idx_to_enum: list[EventType] = []

def init_event_maps():
    if len(event_enum_to_idx) == 0 and len(event_idx_to_enum) == 0:
        for i, x in enumerate(EventType):
            event_enum_to_idx[x] = len(event_idx_to_enum).to_bytes(1)
            event_idx_to_enum.append(x)

def serialize_events(events: list[Event], event_times: list[int]) -> bytearray:
    global event_enum_to_idx
    global event_idx_to_enum
    assert len(events) == len(event_times)
    init_event_maps()

    arr = bytearray(b"\x00" * (4 + len(events) * 5 + len(event_times) * 4))
    view = memoryview(arr)
    pos = 0
    def write(data: bytes | bytearray | memoryview):
        nonlocal pos
        size = len(data)
        view[pos:(pos + size)] = data
        pos += size

    write(len(events).to_bytes(4, "little"))
    for x in events:
        # Sometimes these are actually numpy.int32 and not ints
        write(event_enum_to_idx[x.type])
        write(int(x.value).to_bytes(4, "little", signed=True))
    for x in event_times:
        write(x.to_bytes(4, "little", signed=True))

    assert pos == len(arr)
    return arr

def deserialize_events(data: bytes | bytearray | memoryview) -> tuple[list[Event], list[int]]:
    global event_enum_to_idx
    global event_idx_to_enum
    init_event_maps()

    view = memoryview(data)
    pos = 0
    def read(size: int) -> memoryview:
        nonlocal pos
        ret = view[pos:(pos + size)]
        pos += size
        return ret

    count = int.from_bytes(read(4), "little")

    events: list = [None] * count
    for i in range(count):
        idx = int.from_bytes(read(1))
        time = int.from_bytes(read(4), "little", signed=True)
        events[i] = Event(event_idx_to_enum[idx], time)

    event_times = [0] * count
    for i in range(count):
        event_times[i] = int.from_bytes(read(4), "little", signed=True)

    assert pos == len(data)
    return (events, event_times)
