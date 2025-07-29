import os
import json
import pyzstd
import hydra

from config import TrainConfig
from pathlib import Path
from slider import Beatmap, Circle
from tqdm import tqdm
from dataset.osu_parser import OsuParser
from event import serialize_events
from tokenizer import Tokenizer

BASE_PATH = "E:\\osudataset\\data"

@hydra.main(config_path="configs", config_name="train_v5", version_base="1.1")
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

    def maps_as_bytes():
        for path in beatmap_files:
            with open(path, "r", encoding="utf-8-sig") as beatmap_file:
                beatmap = beatmap_file.read().encode()
            yield beatmap

    '''
    print("training dict")
    zstd_dict = pyzstd.train_dict(maps_as_bytes(), 1024 * 1024)
    with open(f"{BASE_PATH}\\amalgamation.dict", "wb") as amalgamation_dict:
        amalgamation_dict.write(zstd_dict.dict_content)
    '''

    option = {
        pyzstd.CParameter.nbWorkers: 4,
        pyzstd.CParameter.compressionLevel: 5,
    }
    amalgamation_entries = {}
    cur_len = 0
    with open("E:\\osuaigendataset\\amalgamation.bin", "wb") as amalgamation_bin:
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

    with open("E:\\osuaigendataset\\amalgamation.json", "w") as amalgamation_json:
        json.dump(amalgamation_entries, amalgamation_json)

if __name__ == "__main__":
    main()
