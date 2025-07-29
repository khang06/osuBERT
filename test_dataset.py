import hydra

from config import TrainConfig
from dataset.mmrs_dataset import MmrsDataset
from dataset.raw_dataset import RawDataset
from dataset.osu_parser import OsuParser
from tokenizer import Tokenizer
from torch.utils.data import DataLoader

@hydra.main(config_path="configs", config_name="train_v5", version_base="1.1")
def main(args: TrainConfig):
    tokenizer = Tokenizer(args)
    parser = OsuParser(args, tokenizer)
    dataset = MmrsDataset(args=args.data, test=True, parser=parser, tokenizer=tokenizer, shared=None, shuffle=True, mask=False)
    paths = [
        "D:\\osusongs\\440068 Hana - Sakura no Uta\\Hana - Sakura no Uta (Ultimate Madoka) [VI.Artist of the Sakura].osu",
        "D:\\osusongs\\1610294 Mysteka - Hesperos\\Mysteka - Hesperos (Acylica) [3dyoshispin].osu",
    ]
    #dataset = RawDataset(paths, args.data, parser, tokenizer)
    loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
    )

    for x in loader:
        print("data:", x)

if __name__ == "__main__":
    main()
