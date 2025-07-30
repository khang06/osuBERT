import os
import json
import pyzstd
import hydra
import concurrent.futures

from config import TrainConfig
from pathlib import Path
from slider import Beatmap, Circle
from tqdm import tqdm
from dataset.osu_parser import OsuParser
from event import serialize_events
from tokenizer import Tokenizer
from itertools import repeat

def init_process(parser: OsuParser):
    global static_parser
    static_parser = parser

def process_map(path: str) -> None | tuple[bytes, str, float, float, float, float]:
    beatmap = Beatmap.from_path(path)
    if len(beatmap._hit_objects) <= 1 or beatmap.mode != 0:
        return None

    map_start = beatmap._hit_objects[0].time.total_seconds() * 1000
    map_end = (beatmap._hit_objects[-1].time if isinstance(beatmap._hit_objects[-1], Circle) else beatmap._hit_objects[-1].end_time).total_seconds() * 1000
    events, event_times = static_parser.parse(beatmap)

    #beatmap = pyzstd.compress(beatmap, zstd_dict=zstd_dict)
    option = {
        pyzstd.CParameter.nbWorkers: 1,
        pyzstd.CParameter.compressionLevel: 5,
    }
    compressed = pyzstd.compress(serialize_events(events, event_times), option)

    return compressed, str(os.path.basename(path)), map_start, map_end, beatmap.slider_multiplier, beatmap.circle_size

@hydra.main(config_path="configs", config_name="train_v6", version_base="1.1")
def main(args: TrainConfig):
    tokenizer = Tokenizer(args)
    parser = OsuParser(args, tokenizer)

    print("getting files")
    beatmap_files = []
    data_folder = "E:\\osudataset\\data\\data"
    for folder in os.listdir(data_folder):
        beatmap_folder = os.path.join(data_folder, folder)
        for file in os.listdir(beatmap_folder):
            if file.endswith(".osu"):
                beatmap_files.append(Path(os.path.join(beatmap_folder, file)))

    for root, dirs, files in os.walk("E:\\osuaigendataset"):
        for file in files:
            if file.endswith(".osu"):
                beatmap_files.append(os.path.join(root, file))

    option = {
        pyzstd.CParameter.nbWorkers: 4,
        pyzstd.CParameter.compressionLevel: 5,
    }
    amalgamation_entries = {}
    cur_len = 0
    with open("E:\\osuaigendataset\\amalgamation.bin", "wb") as amalgamation_bin:
        '''
        for path in tqdm(beatmap_files):
            try:
                beatmap = Beatmap.from_path(path)
                if len(beatmap._hit_objects) <= 1 or beatmap.mode != 0:
                    continue

                map_start = beatmap._hit_objects[0].time.total_seconds() * 1000
                map_end = (beatmap._hit_objects[-1].time if isinstance(beatmap._hit_objects[-1], Circle) else beatmap._hit_objects[-1].end_time).total_seconds() * 1000
                events, event_times = parser.parse(beatmap)

                #beatmap = pyzstd.compress(beatmap, zstd_dict=zstd_dict)
                compressed = pyzstd.compress(serialize_events(events, event_times), option)
                amalgamation_entries[str(os.path.basename(path))] = (cur_len, len(compressed), map_start, map_end, beatmap.slider_multiplier, beatmap.circle_size)
                cur_len += len(compressed)
                amalgamation_bin.write(compressed)
            except Exception as e:
                print(f"Failed to parse {path}: {e}")
        '''
        with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=init_process, initargs=(parser,)) as executor:
            done = list(tqdm(executor.map(process_map, beatmap_files), total=len(beatmap_files)))
        for x in done:
            if x is None:
                continue
            compressed, filename, map_start, map_end, global_sv, cs = x
            amalgamation_entries[filename] = (cur_len, len(compressed), map_start, map_end, global_sv, cs)
            cur_len += len(compressed)
            amalgamation_bin.write(compressed)

    with open("E:\\osuaigendataset\\amalgamation.json", "w") as amalgamation_json:
        json.dump(amalgamation_entries, amalgamation_json)

if __name__ == "__main__":
    main()
