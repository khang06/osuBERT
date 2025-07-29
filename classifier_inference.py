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

@hydra.main(config_path="configs", config_name="inference_v6", version_base="1.1")
def main(args: TrainConfig):
    tokenizer: Tokenizer = get_tokenizer(args)
    print("vocab size:", tokenizer.vocab_size_in)

    model = ModernBertForSequenceClassification.from_pretrained("../../../export")
    #model = torch.compile(model.eval().half().to("cuda"))
    model = model.eval().to("cuda")

    print("model loaded")
    parser = OsuParser(args, tokenizer)
    test_maps_dir = Path("../../../test_maps/")
    if not test_maps_dir.exists():
        raise FileNotFoundError(f"Test maps directory {test_maps_dir} does not exist.")
    paths = [str(p) for p in test_maps_dir.glob("*.osu")]
    if not paths:
        raise FileNotFoundError(f"No .osu files found in {test_maps_dir}.")
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
                input_ids=x["input_ids"].cuda(), attention_mask=attention_mask
            )
            # print(x)
            probabilities = torch.softmax(features.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1)
            # print(probabilities)
            for i, beatmap_idx in enumerate(x["beatmap_idx"]):
                beatmap_idx = int(beatmap_idx)
                if cur_beatmap != beatmap_idx:
                    print(f"\n{paths[beatmap_idx]}")
                    cur_beatmap = beatmap_idx
                confidence = probabilities[i][predicted_class[i]]
                print(
                    f"{format_timestamp(int(x['start'][i]))} - {format_timestamp(int(x['end'][i]))}: {classes[predicted_class[i]]} ({confidence * 100:.2f}% confident)"
                )

    print("\ndone!!")


if __name__ == "__main__":
    main()
