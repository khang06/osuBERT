import os

beatmap_files = []
for root, dirs, files in os.walk("E:\\osuaigendataset"):
    for file in files:
        if file.endswith(".osu") and not file.endswith("_flip.osu"):
             beatmap_files.append(os.path.join(root, file))

for map_path in beatmap_files:
    print(map_path)
    with open(map_path, "r", encoding="utf-8-sig") as beatmap_file:
        beatmap = beatmap_file.readlines()

    parsing_objs = False
    for i, line in enumerate(beatmap):
        line = line.strip()
        if line.startswith("["):
            parsing_objs = line == "[HitObjects]"
            continue
        if not parsing_objs or len(line) == 0:
            continue

        split = line.split(",")
        split[0] = f"{512 - int(float(split[0]))}"
        if (int(split[3]) & 2) != 0:
            curve_split = split[5].split("|")
            for j, point in enumerate(curve_split):
                if len(point) < 3:
                    continue
                point_split = point.split(":")
                curve_split[j] = f"{512 - int(float(point_split[0]))}:{point_split[1]}"
            split[5] = "|".join(curve_split)
        beatmap[i] = ",".join(split) + "\n"

    with open(map_path[:-4] + "_flip.osu", "w", encoding="utf-8") as output_file:
        output_file.writelines(beatmap)
