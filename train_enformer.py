from dataclasses import dataclass
import random

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser
from Bio import SeqIO
import wandb

from dataloaders.npy_dataset import GEPNpyDataset
from enformer_pytorch import Enformer, EnformerConfig

@dataclass
class ModelArguments:
    model_name_or_path: str = "EleutherAI/enformer-official-rough"


@dataclass
class DataTrainingArguments:
    train_npy_dir: str = "/vepfs-mlp/mlp-public/user/yangzhao/basenji_npy/human_train_targets" # replace with your own train npy dir
    valid_npy_dir: str = "/vepfs-mlp/mlp-public/user/yangzhao/basenji_npy/human_valid_targets" # replace with your own valid npy dir
    train_bed_path: str = "/tos-mlp-zgci/yangzhao/basenji/train_sort_by_h5.bed" # replace with your own train bed path
    valid_bed_path: str = "/tos-mlp-zgci/yangzhao/basenji/valid_sort_by_h5.bed" # replace with your own valid bed path

    genome_path: str = "/tos-mlp-zgci/yangzhao/genome/hg38.ml.fa" # replace with your own genome path
    shift_aug: bool = True
    rc_aug: bool = True
    seqlen: int = 131072

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_wandb_if_main_process(model_args, data_args, training_args):

    if training_args.local_rank == -1:
        wandb.init(
            name=training_args.run_name,
            config={
                "model_args": model_args,
                "data_args": data_args,
                "training_args": training_args,
            }
        )
    else:
        if training_args.local_rank == 0:
            wandb.init(
                name=training_args.run_name,
                config={
                    "model_args": model_args,
                    "data_args": data_args,
                    "training_args": training_args,
                }
            )
            print(f"wandb initialized for DDP training on rank {training_args.local_rank}")
        else:
            print(f"Skip wandb initialization on rank {training_args.local_rank}")

def main():
    # args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"Model arguments: {model_args}")
    print(f"Data arguments: {data_args}")
    # set seed
    set_seed(training_args.seed) 
    # load config 
    if 'wandb' in training_args.report_to:
        init_wandb_if_main_process(model_args, data_args, training_args)

    # import debugpy; debugpy.connect(5678); debugpy.wait_for_client(); debugpy.breakpoint()
    config = EnformerConfig.from_pretrained(model_args.model_name_or_path)

    model = Enformer(config)
    # load genome dict
    genome_dict = SeqIO.to_dict(SeqIO.parse(data_args.genome_path, "fasta"))
    # load dataset
    train_dataset = GEPNpyDataset(
        npy_dir=data_args.train_npy_dir,
        bed_path=data_args.train_bed_path,
        seqlen=data_args.seqlen,
        genome_dict=genome_dict,
        shift_aug=data_args.shift_aug,
        rc_aug=data_args.rc_aug,
    )
    valid_dataset = GEPNpyDataset(
        npy_dir=data_args.valid_npy_dir,
        bed_path=data_args.valid_bed_path,
        seqlen=data_args.seqlen,
        genome_dict=genome_dict,
        shift_aug=False,
        rc_aug=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
