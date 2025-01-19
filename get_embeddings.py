from pathlib import Path
import hydra
import torch

from config import TrainConfig
from dataset.mmrs_dataset import MmrsDataset
from dataset.osu_parser import OsuParser
from model import LitOsuBert, get_tokenizer
from tokenizer import Tokenizer
from transformers import ModernBertModel
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def worker_init_fn(worker_id: int) -> None:
    """
    Give each dataloader a unique slice of the full dataset.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        np.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    )
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

# https://discuss.huggingface.co/t/get-word-embeddings-from-transformer-model/6929/2
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0].float() #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

@hydra.main(config_path="configs", config_name="inference_v4", version_base="1.1")
def main(args: TrainConfig):
    tokenizer: Tokenizer = get_tokenizer(args)
    print("vocab size:", tokenizer.vocab_size_in)

    model: ModernBertModel = LitOsuBert.load_from_checkpoint("../../../final_v4.ckpt").model.model
    model = torch.compile(model.eval().half().to("cuda"))

    parser = OsuParser(args, tokenizer)
    dataset = MmrsDataset(args=args.data, test=True, parser=parser, tokenizer=tokenizer, shuffle=False, mask=False, contiguous=True)
    loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    df = pd.read_parquet(Path(args.data.train_dataset_path) / "metadata.parquet")

    features_tsv = open("../../../features.tsv", "w")
    names_tsv = open("../../../names.tsv", "w")
    names_tsv.write("ID\tName\n")

    # TODO: This is terrible. Rework to use batching properly...
    '''
    cur_beatmap = -1
    cur_id = None
    cur_name = None
    cur_features = []
    for x in loader:
        data_idx = int(x["beatmap_idx"][0])
        if data_idx != cur_beatmap:
            if cur_beatmap != -1 and len(cur_features) > 0:
                np.savetxt(features_tsv, np.reshape(np.mean(cur_features, axis=0), (1, -1)), delimiter="\t")
                names_tsv.write(f"{cur_id}\t{cur_name}\n")
            metadata = df.iloc[data_idx]
            cur_beatmap = data_idx
            cur_id = int(metadata["Id"])
            cur_name = f"{metadata['Artist']} - {metadata['Title']} ({metadata['Creator']}) [{metadata['Version']}]"
            cur_features = []
            print(cur_id, cur_name)

        with torch.no_grad():
            attention_mask = x["attention_mask"].reshape(1, -1).to("cuda")
            features = mean_pooling(model.forward(
                input_ids=x["input_ids"].reshape(1, -1).to("cuda"),
                attention_mask=attention_mask
            ), attention_mask).reshape(-1).cpu()
        cur_features.append(features)
    if cur_beatmap != -1 and len(cur_features) > 0:
        np.savetxt(features_tsv, np.reshape(np.mean(cur_features, axis=0), (1, -1)), delimiter="\t")
        names_tsv.write(f"{cur_id}\t{cur_name}\n")
    '''

    cur_beatmaps: dict[int, list[torch.Tensor]] = {}
    with torch.inference_mode():
        for x in loader:
            attention_mask = x["attention_mask"].cuda()
            features = mean_pooling(model.forward(
                input_ids=x["input_ids"].cuda(),
                attention_mask=attention_mask
            ), attention_mask).cpu()
            print(x["beatmap_idx"])
            queued_deletions = []
            for y in cur_beatmaps.keys():
                if not y in x["beatmap_idx"]:
                    np.savetxt(features_tsv, np.reshape(np.mean(cur_beatmaps[y], axis=0), (1, -1)), delimiter="\t")
                    metadata = df.iloc[y]
                    name = f"{metadata['Artist']} - {metadata['Title']} ({metadata['Creator']}) [{metadata['Version']}]"
                    print(y, name)
                    names_tsv.write(f"{y}\t{name}\n")
                    queued_deletions.append(y)
            for y in queued_deletions:
                del cur_beatmaps[y]
            for i, beatmap_idx in enumerate(x["beatmap_idx"]):
                beatmap_idx = int(beatmap_idx)
                if not beatmap_idx in cur_beatmaps:
                    cur_beatmaps[beatmap_idx] = []
                cur_beatmaps[beatmap_idx].append(features[i])
        for y in cur_beatmaps.keys():
            np.savetxt(features_tsv, np.reshape(np.mean(cur_beatmaps[y], axis=0), (1, -1)), delimiter="\t")
            metadata = df.iloc[y]
            names_tsv.write(f"{y}\t{metadata['Artist']} - {metadata['Title']} ({metadata['Creator']}) [{metadata['Version']}]\n")

    print("done!!")
    features_tsv.close()
    names_tsv.close()

if __name__ == "__main__":
    main()
