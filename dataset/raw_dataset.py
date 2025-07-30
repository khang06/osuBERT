from __future__ import annotations

import os
import random
import math
from multiprocessing.managers import Namespace
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import json
import pyzstd
from pandas import Series, DataFrame
from slider import Beatmap, Circle
from torch.utils.data import IterableDataset

from .data_utils import load_audio_file, remove_events_of_type, get_hold_note_ratio, get_scroll_speed_ratio, \
    get_hitsounded_status, get_song_length
from .osu_parser import OsuParser
from tokenizer import Event, EventType, Tokenizer, ContextType
from config import DataConfig

MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1
LABEL_IGNORE_ID = -100


class RawDataset(IterableDataset):
    def __init__(
            self,
            paths: list[str],
            args: DataConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
    ):
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.paths = paths
        self.class_dropout_prob = args.class_dropout_prob
        self.diff_dropout_prob = args.diff_dropout_prob
        self.add_pre_tokens = args.add_pre_tokens
        self.add_empty_sequences = args.add_empty_sequences

    def _create_sequences(
            self,
            frame_times: npt.NDArray,
            target_context: list[dict],
            extra_data: Optional[dict] = None,
    ) -> list[dict[str, int | npt.NDArray | list[Event] | list[dict]]]:
        """Create frame and token sequences for training/testing.

        Args:
            frames: Audio frames.

        Returns:
            A list of source and target sequences.
        """

        def get_event_indices(events2: list[Event], event_times2: list[int]) -> tuple[list[int], list[int]]:
            if len(events2) == 0:
                return [], []

            # Corresponding start event index for every audio frame.
            start_indices = []
            event_index = 0

            for current_time in frame_times:
                while event_index < len(events2) and event_times2[event_index] < current_time:
                    event_index += 1
                start_indices.append(event_index)

            # Corresponding end event index for every audio frame.
            end_indices = start_indices[1:] + [len(events2)]

            return start_indices, end_indices

        start_indices, end_indices = {}, {}
        for context in target_context:
            (start_indices[context["extra"]["id"]], end_indices[context["extra"]["id"]]) = (
                get_event_indices(context["events"], context["event_times"]))

        sequences = []
        last_kiai = {}
        last_sv = {}
        # Divide audio frames into splits
        for frame_start_idx in range(max(1, len(frame_times) - self.args.overlap_divisor)):
            frame_end_idx = min(frame_start_idx + self.args.overlap_divisor, len(frame_times))

            def slice_events(context, frame_start_idx, frame_end_idx):
                if len(context["events"]) == 0:
                    return []
                identifier = context["extra"]["id"]
                event_start_idx = start_indices[identifier][frame_start_idx]
                event_end_idx = end_indices[identifier][frame_end_idx - 1]
                return context["events"][event_start_idx:event_end_idx]

            def slice_context(context, frame_start_idx, frame_end_idx):
                result = {"events": slice_events(context, frame_start_idx, frame_end_idx)} | context["extra"]
                result["time"] = frame_times[frame_start_idx]
                return result

            # Create the sequence
            sequence: dict[str, str | int | list[Event] | dict] = {
                           "context": [slice_context(context, frame_start_idx, frame_end_idx) for context in target_context],
                       } | extra_data

            sequence["special"] = sequence["special"].copy()
            sequence["special"]["time"] = frame_times[frame_start_idx]

            sequence["start"] = frame_times[frame_start_idx]
            sequence["end"] = frame_times[min(frame_end_idx, len(frame_times) - 1)]

            def add_last_kiai(sequence_context, last_kiai):
                if (sequence_context["context_type"] != ContextType.KIAI and
                        not (self.args.add_kiai and sequence_context["context_type"] in [ContextType.GD, ContextType.MAP])):
                    return
                if sequence_context["id"] in last_kiai:
                    sequence_context["last_kiai"] = last_kiai[sequence_context["id"]]
                else:
                    sequence_context["last_kiai"] = Event(EventType.KIAI, 0)
                # Find the last kiai event in the out context
                for event in reversed(sequence_context["events"]):
                    if event.type == EventType.KIAI:
                        last_kiai[sequence_context["id"]] = event
                        break

            for sequence_context in sequence["context"]:
                add_last_kiai(sequence_context, last_kiai)
                if "last_kiai" in sequence_context:
                    sequence["special"]["last_kiai"] = sequence_context["last_kiai"]

            def add_last_sv(sequence_context, last_sv):
                if (sequence_context["context_type"] != ContextType.SV and
                        not ((self.args.add_sv or self.args.add_mania_sv) and sequence_context["context_type"] in [ContextType.GD, ContextType.MAP])):
                    return
                if sequence_context["id"] in last_sv:
                    sequence_context["last_sv"] = last_sv[sequence_context["id"]]
                else:
                    sequence_context["last_sv"] = Event(EventType.SCROLL_SPEED, 100)
                # Find the last sv event in the out context
                for event in reversed(sequence_context["events"]):
                    if event.type == EventType.SCROLL_SPEED:
                        last_sv[sequence_context["id"]] = event
                        break

            if self.args.add_sv_special_token:
                for sequence_context in sequence["context"]:
                    add_last_sv(sequence_context, last_sv)
                    if "last_sv" in sequence_context:
                        sequence["special"]["last_sv"] = sequence_context["last_sv"]

            sequences.append(sequence)

        return sequences

    def _normalize_time_shifts(self, sequence: dict, beatmap_path) -> dict:
        """Make all time shifts in the sequence relative to the start time of the sequence,
        and normalize time values.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with trimmed time shifts.
        """

        min_t = self.tokenizer.event_range[EventType.TIME_SHIFT].min_value
        max_t = self.tokenizer.event_range[EventType.TIME_SHIFT].max_value

        def process(events: list[Event], start_time) -> list[Event] | tuple[list[Event], int]:
            for i, event in enumerate(events):
                if event.type == EventType.TIME_SHIFT:
                    # We cant modify the event objects themselves because that will affect subsequent sequences
                    t = int((event.value - start_time) * STEPS_PER_MILLISECOND)
                    if t < min_t or t > max_t:
                        print(f"WARNING: Time shift out of range ({t}) in beatmap {beatmap_path}")
                        t = np.clip(t, min_t, max_t)
                    events[i] = Event(EventType.TIME_SHIFT, t)

            return events

        if "pre_events" in sequence:
            sequence["pre_events"] = process(sequence["pre_events"], sequence["context"]["time"])

        for context in sequence["context"]:
            context["events"] = process(context["events"], context["time"])

        return sequence

    def _get_special_token_count(self, context: dict) -> int:
        predicates = [
            self.args.add_gamemode_token,
            self.args.add_style_token,
            self.args.add_diff_token,
            self.args.add_mapper_token,
            self.args.add_year_token,
            self.args.add_hitsounded_token,
            self.args.add_song_length_token,
            self.args.add_sv and "global_sv" in context,
            self.args.add_cs_token and "circle_size" in context,
            "last_kiai" in context,
            "last_sv" in context,
        ]
        sum = 1
        for x in predicates:
            if x:
                sum += 1
        return sum

    def _get_special_tokens(self, context: dict) -> list:
        special_tokens = []

        if "beatmap_id" in context:
            if self.args.add_gamemode_token:
                special_tokens.append(self.tokenizer.encode_gamemode(context["gamemode"]))

            if self.args.add_style_token:
                special_tokens.append(self.tokenizer.encode_style_idx(context["beatmap_idx"])
                                      if random.random() >= self.args.class_dropout_prob else self.tokenizer.style_unk)

            if self.args.add_diff_token:
                special_tokens.append(self.tokenizer.encode_diff(context["difficulty"])
                                      if random.random() >= self.args.diff_dropout_prob else self.tokenizer.diff_unk)

            if self.args.add_mapper_token:
                special_tokens.append(self.tokenizer.encode_mapper(context["beatmap_id"])
                                      if random.random() >= self.args.mapper_dropout_prob else self.tokenizer.mapper_unk)

            if self.args.add_year_token:
                special_tokens.append(self.tokenizer.encode_year(context["year"])
                                      if random.random() >= self.args.year_dropout_prob else self.tokenizer.year_unk)

            if self.args.add_hitsounded_token:
                special_tokens.append(self.tokenizer.encode(Event(EventType.HITSOUNDED, int(context["hitsounded"]))))

            if self.args.add_song_length_token:
                special_tokens.append(self.tokenizer.encode_song_length(context["song_length"]))

            if self.args.add_sv and "global_sv" in context:
                special_tokens.append(self.tokenizer.encode_global_sv(context["global_sv"]))

            if self.args.add_cs_token and "circle_size" in context:
                special_tokens.append(self.tokenizer.encode_cs(context["circle_size"])
                                      if random.random() >= self.args.cs_dropout_prob else self.tokenizer.cs_unk)


            # TODO: I don't feel like implementing these right now
            #if "keycount" in context:
            #    special_tokens.append(self.tokenizer.encode(Event(EventType.MANIA_KEYCOUNT, context["keycount"])))

            #if "hold_note_ratio" in context:
            #    special_tokens.append(self.tokenizer.encode_hold_note_ratio(context["hold_note_ratio"])
            #                          if random.random() >= self.args.hold_note_ratio_dropout_prob else self.tokenizer.hold_note_ratio_unk)

            #if "scroll_speed_ratio" in context:
            #    special_tokens.append(self.tokenizer.encode_scroll_speed_ratio(context["scroll_speed_ratio"])
            #                          if random.random() >= self.args.scroll_speed_ratio_dropout_prob else self.tokenizer.scroll_speed_ratio_unk)

            #if self.args.add_descriptors:
            #    special_tokens.extend(self.tokenizer.encode_descriptor(context["beatmap_id"])
            #                          if random.random() >= self.args.descriptor_dropout_prob else [
            #        self.tokenizer.descriptor_unk])

            if "last_kiai" in context:
                special_tokens.append(self.tokenizer.encode(context["last_kiai"]))

            if "last_sv" in context:
                special_tokens.append(self.tokenizer.encode(context["last_sv"]))

            if self.args.add_song_position_token:
                special_tokens.append(self.tokenizer.encode_song_position(context["time"], context["song_length"]))

        return special_tokens

    def _tokenize_sequence(self, sequence: dict) -> dict:
        """Tokenize the event sequence.

        Begin token sequence with `[SOS]` token (start-of-sequence).
        End token sequence with `[EOS]` token (end-of-sequence).

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with tokenized events.
        """
        #sequence["special_tokens"] = self._get_special_tokens(sequence["special"])
        sequence["special_tokens"] = []

        for context in sequence["context"]:
            tokens = torch.empty(len(context["events"]), dtype=torch.long)
            for i, event in enumerate(context["events"]):
                tokens[i] = self.tokenizer.encode(event)
            context["tokens"] = tokens
            context["special_tokens"] = self._get_special_tokens(context)

        if "pre_events" in sequence:
            pre_tokens = torch.empty(len(sequence["pre_events"]), dtype=torch.long)
            for i, event in enumerate(sequence["pre_events"]):
                pre_tokens[i] = self.tokenizer.encode(event)
            sequence["pre_tokens"] = pre_tokens
            del sequence["pre_events"]

        return sequence

    def _pad_and_split_token_sequence(self, sequence: dict) -> tuple[dict, int]:
        """Pad token sequence to a fixed length and split decoder input and labels.

        Pad with `[PAD]` tokens until `tgt_seq_len`.

        Token sequence (w/o last token) is the input to the transformer decoder,
        token sequence (w/o first token) is the label, a.k.a. decoder ground truth.

        Prefix the token sequence with the pre_tokens sequence.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded tokens.
        """
        # Count irreducable tokens for SOS/EOS tokens
        stl = 1

        # Count irreducable tokens for all contexts
        stl += len(sequence["special_tokens"])
        for context in sequence["context"]:
            if context["add_type"]:
                stl += 2

            stl += len(context["special_tokens"])

        # Count reducible tokens, pre_tokens and context tokens
        num_tokens = sum(len(context["tokens"]) for context in sequence["context"])

        # Trim tokens to target sequence length
        # n + stl + padding = tgt_seq_len
        n = min(self.args.tgt_seq_len - stl, num_tokens)
        si = 0

        input_tokens = torch.full((self.args.tgt_seq_len,), self.tokenizer.pad_id, dtype=torch.long)
        label_tokens = torch.full((self.args.tgt_seq_len,), LABEL_IGNORE_ID, dtype=torch.long)

        def add_special_tokens(special_tokens, si):
            for token in special_tokens:
                input_tokens[si] = token
                si += 1
            return si

        def add_context(context, si, max_tokens):
            if context["add_type"]:
                input_tokens[si] = self.tokenizer.context_sos[context["context_type"]]
                si += 1

            si = add_special_tokens(context["special_tokens"], si)

            num_other_tokens_to_add = min(len(context["tokens"]), max_tokens)
            input_tokens[si:si + num_other_tokens_to_add] = context["tokens"][:num_other_tokens_to_add]
            si += num_other_tokens_to_add
            max_tokens -= num_other_tokens_to_add

            if context["add_type"]:
                input_tokens[si] = self.tokenizer.context_eos[context["context_type"]]
                si += 1

            return si, max_tokens

        input_tokens[si] = self.tokenizer.cls_id
        si += 1

        si = add_special_tokens(sequence["special_tokens"], si)
        start_random_index = si

        start_label_index = si
        for context in sequence["context"]:
            si, n = add_context(context, si, n)
        end_index = si

        # Randomize some input tokens
        def randomize_tokens(tokens):
            offset = torch.randint(low=-self.args.timing_random_offset, high=self.args.timing_random_offset + 1,
                                   size=tokens.shape)
            return torch.where((self.tokenizer.event_start[EventType.TIME_SHIFT] <= tokens) & (
                    tokens < self.tokenizer.event_end[EventType.TIME_SHIFT]),
                               torch.clamp(tokens + offset,
                                           self.tokenizer.event_start[EventType.TIME_SHIFT],
                                           self.tokenizer.event_end[EventType.TIME_SHIFT] - 1),
                               tokens)

        if self.args.timing_random_offset > 0:
            input_tokens[start_random_index:end_index] = randomize_tokens(input_tokens[start_random_index:end_index])
        # input_tokens = torch.where((self.tokenizer.event_start[EventType.DISTANCE] <= input_tokens) & (input_tokens < self.tokenizer.event_end[EventType.DISTANCE]),
        #                               torch.clamp(input_tokens + torch.randint_like(input_tokens, -10, 10), self.tokenizer.event_start[EventType.DISTANCE], self.tokenizer.event_end[EventType.DISTANCE] - 1),
        #                               input_tokens)



        #label_tokens[:end_index] = input_tokens[:end_index]

        sequence["input_ids"] = input_tokens
        sequence["attention_mask"] = input_tokens != self.tokenizer.pad_id

        del sequence["context"]
        del sequence["special_tokens"]
        del sequence["special"]
        if "pre_tokens" in sequence:
            del sequence["pre_tokens"]

        return (sequence, end_index)

    def __iter__(self):
        return self._get_next_tracks()

    def _get_next_tracks(self) -> dict:
        for i, path in enumerate(self.paths):
            for sample in self._get_next_beatmap(i, path, 1.0):
                yield sample

    def _get_next_beatmap(self, i, path: str, speed: float) -> dict:
        osu_beatmap = Beatmap.from_path(path)
        if len(osu_beatmap._hit_objects) <= 1:
            return

        map_start = osu_beatmap._hit_objects[0].time.total_seconds() * 1000
        map_end = (osu_beatmap._hit_objects[-1].time if isinstance(osu_beatmap._hit_objects[-1], Circle) else osu_beatmap._hit_objects[-1].end_time).total_seconds() * 1000
        frame_times = np.append(np.arange(map_start, map_end, self.args.ms_per_seq // self.args.overlap_divisor), map_end)

        def get_context(context: ContextType, identifier, add_type=True):
            data = {"extra": {"context_type": context, "add_type": add_type, "id": identifier + '_' + context.value}}
            if context == ContextType.NONE:
                data["events"], data["event_times"] = [], []
            elif context == ContextType.TIMING:
                data["events"], data["event_times"] = self.parser.parse_timing(osu_beatmap, speed)
            elif context == ContextType.NO_HS:
                hs_events, hs_event_times = self.parser.parse(osu_beatmap, speed)
                data["events"], data["event_times"] = remove_events_of_type(hs_events, hs_event_times,
                                                                            [EventType.HITSOUND, EventType.VOLUME])
            elif context == ContextType.MAP:
                data["events"], data["event_times"] = self.parser.parse(osu_beatmap, speed)
                if random.random() < self.args.hitsound_dropout_prob:
                    data["events"], data["event_times"] = remove_events_of_type(data["events"], data["event_times"],
                                                                                [EventType.HITSOUND, EventType.VOLUME])
            elif context == ContextType.KIAI:
                data["events"], data["event_times"] = self.parser.parse_kiai(osu_beatmap, speed)
            elif context == ContextType.SV:
                data["events"], data["event_times"] = self.parser.parse_scroll_speeds(osu_beatmap, speed)
            return data

        extra_data = {
            "beatmap_idx": i,
            "special": {},
        }

        context = [get_context(context, "out", add_type=False) for context in [ContextType.MAP]]
        sequences = self._create_sequences(frame_times, context, extra_data)

        for sequence in sequences:
            sequence = self._normalize_time_shifts(sequence, path)
            sequence = self._tokenize_sequence(sequence)
            (sequence, seq_len) = self._pad_and_split_token_sequence(sequence)
            #if self.mask and not self.add_empty_sequences and ((sequence["labels"] == self.tokenizer.cls_id) | (
            #        sequence["labels"] == LABEL_IGNORE_ID)).all():
            #    continue

            if not self.add_empty_sequences:
                if seq_len <= 1:
                    print("lol continuing", sequence, seq_len)
                    continue

            # if sequence["decoder_input_ids"][self.pre_token_len - 1] != self.tokenizer.pad_id:
            #     continue
            #sequence["mapper_labels"] = self.tokenizer.mapper_idx[self.tokenizer.beatmap_mapper[osu_beatmap.beatmap_id]]

            yield sequence
