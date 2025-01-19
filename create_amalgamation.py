import os
import json
import pyzstd
from pathlib import Path
from tqdm import tqdm

BASE_PATH = "E:\\osudataset\\data"

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
        with open(path, "r", encoding="utf-8-sig") as beatmap_file:
            beatmap = beatmap_file.read().encode()
        #beatmap = pyzstd.compress(beatmap, zstd_dict=zstd_dict)
        beatmap = pyzstd.compress(beatmap, option)
        amalgamation_entries[str(os.path.basename(path))] = (cur_len, len(beatmap))
        cur_len += len(beatmap)
        amalgamation_bin.write(beatmap)
with open("E:\\osuaigendataset\\amalgamation.json", "w") as amalgamation_json:
    json.dump(amalgamation_entries, amalgamation_json)
