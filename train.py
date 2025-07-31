import hydra
import lightning
import torch
import numpy as np
from config import TrainConfig
from dataset.mmrs_dataset import MmrsDataset
from torch.utils.data import DataLoader, Dataset
from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertForSequenceClassification
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, OnExceptionCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dataset.osu_parser import OsuParser
from model import LitOsuBert, LitOsuBertClassifier, get_tokenizer
from tokenizer import Tokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def get_dataset(args: TrainConfig, **kwargs) -> Dataset:
    if args.data.dataset_type == "mmrs":
        return MmrsDataset(args=args.data, **kwargs)
    else:
        raise NotImplementedError

def get_dataloaders(tokenizer: Tokenizer, args: TrainConfig) -> tuple[DataLoader, DataLoader]:
    parser = OsuParser(args, tokenizer)
    dataset = {
        "train": get_dataset(
            args=args,
            test=False,
            parser=parser,
            tokenizer=tokenizer,
            shared=None,
            mask=args.task == "mask",
        ),
        "test": get_dataset(
            args=args,
            test=True,
            parser=parser,
            tokenizer=tokenizer,
            shared=None,
            mask=args.task == "mask",
        ),
    }

    dataloaders = {}
    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=args.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.dataloader.num_workers > 0,
            worker_init_fn=worker_init_fn,
        )

    return dataloaders["train"], dataloaders["test"]

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

@hydra.main(config_path="configs", config_name="train_v7_ai", version_base="1.1")
def main(args: TrainConfig):
    match args.task:
        case "mask":
            task_project = "osuBERT"
        case "ai":
            task_project = "detectoratorinator"
        case x:
            raise Exception(f"Unhandled task {x}")

    wandb_logger = WandbLogger(
        project=task_project,
        entity="khangarood",
        job_type="training",
        offline=args.logging.mode == "offline",
        log_model="all" if args.logging.mode == "online" else False,
    )

    tokenizer: Tokenizer = get_tokenizer(args)
    print("vocab size:", tokenizer.vocab_size_in)

    train_dataloader, val_dataloader = get_dataloaders(tokenizer, args)

    match args.task:
        case "mask":
            model = LitOsuBert(args, tokenizer)
        case "ai":
            model = LitOsuBertClassifier(args, tokenizer)
            if args.pretrained_path != "":
                print(f"loading pretrained model {args.pretrained_path}")
                model.model.model = ModernBertForMaskedLM.from_pretrained(args.pretrained_path).model
        case x:
            raise Exception(f"Unhandled task {x}")
    #model.model.gradient_checkpointing_enable()

    if args.compile:
        model.model = torch.compile(model.model)

    checkpoint_callback = ModelCheckpoint(every_n_train_steps=args.checkpoint.every_steps, save_top_k=2, monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    exception_checkpoint = OnExceptionCheckpoint(".")
    trainer = lightning.Trainer(
        accelerator=args.device,
        precision=args.precision,
        logger=wandb_logger,
        max_steps=args.optim.total_steps,
        #max_epochs=10,
        accumulate_grad_batches=args.optim.grad_acc,
        gradient_clip_val=args.optim.grad_clip,
        val_check_interval=args.eval.every_steps,
        log_every_n_steps=args.logging.every_steps,
        callbacks=[checkpoint_callback, lr_monitor, exception_checkpoint],

        # DEBUG
        #profiler="simple",
        #max_epochs=1,
        limit_val_batches=args.eval.steps,
        #detect_anomaly=True
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("final.ckpt")

if __name__ == "__main__":
    main()
