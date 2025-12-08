"""
Training script for IndexTTS GPT finetuning on Stress17K dataset
Uses pre-computed w2v-BERT features for efficient training
"""
import os
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import argparse

from pl_module import IndexTTSLightningModulePrecomputed
from precomputed_dataset import create_precomputed_dataloader


def train(config_path: str):
    """Main training function"""
    
    # Load configuration
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from: {config_path}")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Create dataloaders
    print("\n" + "="*60)
    print("CREATING DATALOADERS FOR STRESS17K")
    print("="*60)
    
    train_loader = create_precomputed_dataloader(
        features_roots=cfg.data.features_roots,
        metadata_paths=cfg.data.train_metadata_paths,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        split="train",
        model_dir=cfg.model_dir,
        max_audio_length=cfg.data.max_audio_length,
        max_text_length=cfg.data.max_text_length,
    )
    
    val_loader = create_precomputed_dataloader(
        features_roots=cfg.data.features_roots,
        metadata_paths=cfg.data.val_metadata_paths,
        batch_size=cfg.data.val_batch_size,
        num_workers=cfg.data.num_workers,
        split="val",
        model_dir=cfg.model_dir,
        max_audio_length=cfg.data.max_audio_length,
        max_text_length=cfg.data.max_text_length,
    )
    
    print(f"\n✅ Train dataloader: {len(train_loader)} batches")
    print(f"✅ Val dataloader: {len(val_loader)} batches")

    # raise NotImplementedError("Do not use this script. Use train_paraspeechcaps.py instead.")
    
    # Initialize Lightning module
    print("\n" + "="*60)
    print("INITIALIZING LIGHTNING MODULE")
    print("="*60)
    
    model = IndexTTSLightningModulePrecomputed(
        finetune_config_path=config_path
    )
    
    print("✅ Lightning module initialized")
    
    # Setup logging
    logger = None
    if cfg.logging.use_wandb:
        logger = WandbLogger(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.logging.experiment_name,
            tags=cfg.logging.wandb_tags,
            notes=cfg.logging.wandb_notes,
            save_dir=cfg.logging.output_dir,
        )
        print(f"✅ WandB logger initialized (project: {cfg.logging.wandb_project})")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging.output_dir, "checkpoints"),
        filename="stress17k-{epoch:02d}-{val_loss:.4f}",
        save_top_k=cfg.logging.save_top_k,
        monitor="val_loss",
        mode="min",
        save_last=True,
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    callbacks = [checkpoint_callback, lr_monitor]
    print(f"✅ Callbacks configured (saving top {cfg.logging.save_top_k} checkpoints)")
    
    # Setup trainer
    print("\n" + "="*60)
    print("CONFIGURING TRAINER")
    print("="*60)
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.hardware.gpus,
        precision=cfg.hardware.precision,
        accumulate_grad_batches=cfg.hardware.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy="ddp" if cfg.hardware.gpus > 1 else "auto",
        sync_batchnorm=True if cfg.hardware.gpus > 1 else False,
    )

    if trainer.is_global_zero:
        # Create output directory
        run_directory = cfg.logging.output_dir
        os.makedirs(run_directory, exist_ok=True)
        # copy config to output dir
        OmegaConf.save(cfg, os.path.join(run_directory, "config.yaml"))
        # redirect stdout to log file
        log_file = os.path.join(run_directory, "training.log")
        logging.basicConfig(filename=log_file, level=logging.INFO)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
        logging.info(f"Experiment run directory: {run_directory}")
    
    print(f"✅ Trainer configured:")
    print(f"   - Max epochs: {cfg.training.max_epochs}")
    print(f"   - Max steps: {cfg.training.max_steps}")
    print(f"   - GPUs: {cfg.hardware.gpus}")
    print(f"   - Precision: {cfg.hardware.precision}")
    print(f"   - Gradient accumulation: {cfg.hardware.accumulate_grad_batches}")
    print(f"   - Gradient clipping: {cfg.training.gradient_clip_val}")
    
    # Resume from checkpoint if specified
    resume_ckpt = cfg.resume.checkpoint_path if cfg.resume.checkpoint_path else None
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"\n⚠️  Resuming from checkpoint: {resume_ckpt}")
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING ON STRESS17K DATASET")
    print("="*60 + "\n")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train IndexTTS GPT on Stress17K dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/finetune_stress17k_config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print(f"\n{'='*60}")
    print(f"STRESS17K TRAINING SCRIPT")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"{'='*60}\n")
    
    # Start training
    train(args.config)


if __name__ == "__main__":
    main()
