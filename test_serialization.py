import hydra
from slider import Beatmap

from config import TrainConfig
from dataset.mmrs_dataset import MmrsDataset
from dataset.raw_dataset import RawDataset
from dataset.osu_parser import OsuParser
from event import serialize_events, deserialize_events
from tokenizer import Tokenizer

@hydra.main(config_path="configs", config_name="train_v5", version_base="1.1")
def main(args: TrainConfig):
    tokenizer = Tokenizer(args)
    parser = OsuParser(args, tokenizer)

    print("parsing")
    beatmap = Beatmap.from_path("D:\\osusongs\\724034\\Various Artists - Alternator Compilation (Monstrata) [Marathon].osu")
    events, event_times = parser.parse(beatmap)
    print(len(events), len(event_times))
    print(events[:16], event_times[:16])
    print(events[-16:], event_times[-16:])

    print("serializing")
    serialized = serialize_events(events, event_times)
    print(len(serialized))

    print("deserializing")
    events, event_times = deserialize_events(serialized)
    print(len(events), len(event_times))
    print(events[:16], event_times[:16])
    print(events[-16:], event_times[-16:])

if __name__ == "__main__":
    main()
