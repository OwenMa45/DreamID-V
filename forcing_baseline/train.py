"""Top-level training entry. Dispatches on ``config.trainer``:

  * ``diffusion``                -> Stage-1 causal AR diffusion (teacher forcing)
  * ``consistency_distillation`` -> Stage-2 causal Consistency Distillation
  * ``distillation``             -> Stage-3 DMD

Launch with torchrun, e.g.::

    torchrun --nproc_per_node=8 train.py --config_path configs/ar_diffusion.yaml \
        --logdir checkpoints/chunkwise/stage1_ar --disable-wandb
"""
import argparse
import os

from omegaconf import OmegaConf


def build_trainer(config):
    if config.trainer == "diffusion":
        from trainer.diffusion import Trainer
    elif config.trainer == "consistency_distillation":
        from trainer.naive_cd import Trainer
    elif config.trainer == "distillation":
        from trainer.distillation import Trainer
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    return Trainer(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--wandb-save-dir", type=str, default="")
    parser.add_argument("--disable-wandb", action="store_true")
    # Resume control (Stage-1 diffusion trainer): point at a specific checkpoint
    # dir / model.pt, or let it auto-resume from the latest ckpt in --logdir.
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--no-auto-resume", action="store_true",
                        help="ignore existing checkpoints in --logdir and start fresh")
    # Init override: replaces config.generator_ckpt without editing the yaml
    # (e.g. Stage-3a DMD initialised from a chosen Stage-2 checkpoint).
    parser.add_argument("--generator-ckpt", type=str, default="",
                        help="override config.generator_ckpt (path to a model.pt)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.merge(
        OmegaConf.load(os.path.join(here, "configs", "default_config.yaml")),
        OmegaConf.load(args.config_path))
    config.no_save = args.no_save
    config.config_name = os.path.basename(args.config_path).split(".")[0]
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    if args.disable_wandb:
        config.disable_wandb = True
    if args.resume_ckpt:
        config.resume_ckpt = args.resume_ckpt
    if args.no_auto_resume:
        config.auto_resume = False
    if args.generator_ckpt:
        config.generator_ckpt = args.generator_ckpt

    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
