import hydra
import torch
import numpy as np

from config import TrainConfig
from dataset.mmrs_dataset import MmrsDataset
from dataset.raw_dataset import RawDataset
from dataset.osu_parser import OsuParser
from tokenizer import Tokenizer
from torch.utils.data import DataLoader

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

@hydra.main(config_path="configs", config_name="train_v6", version_base="1.1")
def main(args: TrainConfig):
    tokenizer = Tokenizer(args)
    parser = OsuParser(args, tokenizer)
    dataset = MmrsDataset(args=args.data, test=False, parser=parser, tokenizer=tokenizer, shared=None, shuffle=True, mask=False)
    paths = [
        "D:\\osusongs\\440068 Hana - Sakura no Uta\\Hana - Sakura no Uta (Ultimate Madoka) [VI.Artist of the Sakura].osu",
        "D:\\osusongs\\1610294 Mysteka - Hesperos\\Mysteka - Hesperos (Acylica) [3dyoshispin].osu",
    ]
    #dataset = RawDataset(paths, args.data, parser, tokenizer)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.optim.batch_size // args.optim.grad_acc,
        num_workers=args.dataloader.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.dataloader.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    for i, x in enumerate(loader):
        #print("data:", x)
        print(i)

if __name__ == "__main__":
    main()
