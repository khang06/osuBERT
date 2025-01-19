from pathlib import Path
import hydra
import torch

from config import TrainConfig
from dataset.raw_dataset import RawDataset
from dataset.osu_parser import OsuParser
from model import LitOsuBertClassifier, get_tokenizer
from tokenizer import Tokenizer
from transformers import ModernBertForSequenceClassification
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def format_timestamp(t: int):
    return f"{t // 60000:02}:{(t // 1000) % 60:02}:{t % 1000:03}"

@hydra.main(config_path="configs", config_name="inference_v5", version_base="1.1")
def main(args: TrainConfig):
    tokenizer: Tokenizer = get_tokenizer(args)
    print("vocab size:", tokenizer.vocab_size_in)

    # Requires TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for now. Sorry
    model: ModernBertForSequenceClassification = LitOsuBertClassifier.load_from_checkpoint("../../../final_v5.ckpt").model
    #model = torch.compile(model.eval().half().to("cuda"))
    model = model.eval().to("cuda")

    print("model loaded")
    parser = OsuParser(args, tokenizer)
    paths = [
        "D:\\osusongs\\beatmap-638579819698122505-ultimatedragonforestkingdom64\\KEMOMIMI EDM SQUAD - ultimate dragon forest kingdom 64 definitive edition (game of the year mix) (Mapperatorinator) [Mapperatorinator V30].osu",
        "D:\\osusongs\\beatmap-638414025520738565-lightning\\PSYQUI - Lightning (Mapperatorinator) [Mapperatorinator V30].osu",
        "D:\\osusongs\\974908 Kaytie - NOT THE YADDAS _piano version_ [no video]\\beatmap21a1e2b924404de6af593e3b25a9d00d.osu",
        "D:\\osusongs\\beatmap-638307673014842745-boot\\paraoka - boot (Mapperatorinator) [Mapperatorinator V30].osu",
        "D:\\osusongs\\1471137 katagiri - Angel's Salad [no video]\\katagiri - Angel's Salad (Mapperatorinator) [Mapperatorinator V30].osu",
        "D:\\osusongs\\beatmap-638752034725044293-herheadissoooorolling\\leroy - her head is soooo rolling!! love her (Mapperatorinator) [Mapperatorinator V29.1].osu",
        "D:\\osusongs\\1719345 sasakureUK - Decadence feat mami\\sasakure.UK - Decadence feat. mami (Mapperatorinator) [Mapperatorinator V29.1].osu",
        "D:\\osusongs\\beatmap-637965389806714115-25. Iron Maiden Strength - When I'm Dead\\Iron Maiden Strength - When I'm Dead (Mapperatorinator) [Mapperatorinator V29.1].osu",
        "D:\\osusongs\\beatmap-638245921147812986-dorchadas\\Rita - DORCHADAS (Mapperatorinator) [Mapperatorinator V29].osu",
        "D:\\osusongs\\422136\\Sota Fujimori - polygon (Mapperatorinator) [Mapperatorinator V29].osu",
        "D:\\osusongs\\531488\\Nanahoshi Kangengakudan feat. Hatsune Miku - No.39 (Mapperatorinator) [Mapperatorinator V29].osu",
        "E:\\osuaigendataset\\data\\sakuraburst - SHA (handsome) [Diffusion 2662928 0].osu",
        "E:\\osuaigendataset\\data\\Tatsh - IMAGE -MATERIAL- Version 0 (Scorpiour) [Diffusion 900957 0].osu",
        "E:\\osuaigendataset\\data\\Memme - Chinese Restaurant (rrtyui) [Diffusion None 0 2023-09-16 142411.581370].osu",
        "D:\\osusongs\\90935 IOSYS - Endless Tewi-ma Park\\IOSYS - Endless Tewi-ma Park (Lanturn) [Tewi 2B Expert Edition].osu",
        "D:\\chasermaps\\No Good - Kito Yak Dulu Lagi (KYDL) (chaser01) [melayu apabila melayu yang lain berjaya_].osu",
        "D:\\chasermaps\\Annabel - historia (Game Ver.) (chaser01) [Extra] (3).osu",
        "D:\\chasermaps\\mao - Kimi to Issho ni (chaser01) [together with you].osu",
        "D:\\chasermaps\\Various Artists - UKF Drum & Bass 2010 (Album Megamix) (chaser01) [Marathon] (2).osu",
        "D:\\chasermaps\\anNina - taishou a ~adonde vuelvo~ (chaser01) [inNsane].osu",
        "D:\\chasermaps\\Plastic Tree - Kuchizuke (chaser01) [Relic] (2).osu",
        "D:\\chasermaps\\niki - Close to you (chaser01) [HanzeR's ;_;].osu",
        "D:\\chasermaps\\Kano Nanaka - Sakura to Himawari (chaser01) [smile].osu",
        "D:\\chasermaps\\Hondo Kaede, Oonishi Saori, Waki Azumi, Kino Hina, Matsuda Risae, Suzuki Eri - Save you Save me (chaser01) [Ex].osu",
        "D:\\chasermaps\\Brookes Brothers - Tear You Down (chaser01) [Challenge].osu",
        "D:\\chasermaps\\Tamura Yukari - Beautiful Amulet (chaser01) [Ex].osu",
    ]
    dataset = RawDataset(paths, args.data, parser, tokenizer)
    loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
    )

    # TODO: Make this configurable
    classes = [
        "Not AI",
        "AI????",
    ]

    cur_beatmap = -1
    with torch.inference_mode():
        for x in loader:
            attention_mask = x["attention_mask"].cuda()
            features = model.forward(
                input_ids=x["input_ids"].cuda(),
                attention_mask=attention_mask
            )
            #print(x)
            probabilities = torch.softmax(features.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1)
            #print(probabilities)
            for i, beatmap_idx in enumerate(x["beatmap_idx"]):
                beatmap_idx = int(beatmap_idx)
                if cur_beatmap != beatmap_idx:
                    print(f"\n{paths[beatmap_idx]}")
                    cur_beatmap = beatmap_idx
                confidence = probabilities[i][predicted_class[i]]
                print(f"{format_timestamp(int(x['start'][i]))} - {format_timestamp(int(x['end'][i]))}: {classes[predicted_class[i]]} ({confidence * 100:.2f}% confident)")

    print("\ndone!!")

if __name__ == "__main__":
    main()
