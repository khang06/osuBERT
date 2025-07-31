import lightning
from omegaconf import DictConfig

from config import TrainConfig
from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertForSequenceClassification
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ConstantLR,
)
import bitsandbytes as bnb

from tokenizer import Tokenizer

class LitOsuBert(lightning.LightningModule):
    def __init__(self, args: TrainConfig, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        config = ModernBertConfig(
            vocab_size=tokenizer.vocab_size_in,
            num_hidden_layers=16,
            num_attention_heads=12,
            hidden_size=384,
            pad_token_id=tokenizer.pad_id,
            cls_token_id=tokenizer.cls_id,
        )
        self.model = ModernBertForMaskedLM(config)

    def forward(self, **kwargs) -> MaskedLMOutput:
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        output: MaskedLMOutput = self.model(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def testy_step(self, batch, batch_idx, prefix):
        output: MaskedLMOutput = self.model(**batch)
        loss = output.loss
        self.log(f"{prefix}_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.testy_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.testy_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = get_optimizer(self, self.args)
        scheduler = get_scheduler(optimizer, self.args)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }}

class LitOsuBertClassifier(lightning.LightningModule):
    def __init__(self, args: TrainConfig, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        config = ModernBertConfig(
            vocab_size=tokenizer.vocab_size_in,
            num_hidden_layers=16,
            num_attention_heads=12,
            hidden_size=384,
            pad_token_id=tokenizer.pad_id,
            cls_token_id=tokenizer.cls_id,
            classifier_dropout=0.1,
        )
        # TODO: Make this configurable
        config.num_labels = 2
        self.model = ModernBertForSequenceClassification(config)

    def forward(self, **kwargs) -> SequenceClassifierOutput:
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        output: SequenceClassifierOutput = self.model(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def testy_step(self, batch, batch_idx, prefix):
        output: SequenceClassifierOutput = self.model(**batch)
        loss = output.loss
        self.log(f"{prefix}_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.testy_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.testy_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = get_optimizer(self, self.args)
        scheduler = get_scheduler(optimizer, self.args)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }}

# TODO: These are definitely not the right parameters to use
def get_optimizer(model: lightning.LightningModule, args: DictConfig) -> Optimizer:
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == 'adamw':
        '''
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
        '''
        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'muon':
        from muon_utils import Muon
        """
        Muon is intended to optimize only the internal ≥2D parameters of a network.
        Embeddings, classifier heads, and scalar or vector parameters should be optimized using AdamW.
        """
        adamw_params = [
            param for name, param in model.named_parameters()
            if (any(kw in name.lower() for kw in {'embed', 'proj_out'}) or param.ndim <= 1)
        ]

        adamw_param_set = set(adamw_params)
        muon_params = [
            param for _, param in model.named_parameters()
            if param not in adamw_param_set
        ]
        print(f"Number of parameters for Muon: {len(muon_params)}")
        print(f"Number of parameters for AdamW: {len(adamw_params)}")

        optimizer = Muon(
            muon_params=muon_params,
            lr=args.optim.base_lr,
            adamw_lr=args.optim.base_lr_2,
            adamw_params=adamw_params,
            adamw_betas=(0.90, 0.95),
            adamw_wd=args.optim.weight_decay,
        )
    else:
        print(args.optim.name)
        raise NotImplementedError

    return optimizer

def get_scheduler(optimizer: Optimizer, args: TrainConfig) -> LRScheduler:
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps - args.optim.warmup_steps,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps],
    )

    return scheduler

def get_tokenizer(args: TrainConfig) -> Tokenizer:
    return Tokenizer(args)
