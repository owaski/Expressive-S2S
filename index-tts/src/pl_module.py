"""
PyTorch Lightning Module for IndexTTS GPT Finetuning with Pre-computed Features
This version is specifically designed to work with pre-computed w2v-BERT features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from typing import Dict, Any, Tuple
import os
from huggingface_hub import hf_hub_download
import safetensors
import yaml

from indextts.gpt.model_v2 import UnifiedVoice, build_hf_gpt_transformer
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.s2mel.modules.audio import mel_spectrogram

def calculate_num_params(model: nn.Module) -> int:
    """Calculate the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class IndexTTSLightningModulePrecomputed(pl.LightningModule):
    """
    PyTorch Lightning module for finetuning IndexTTS GPT model (text-to-semantic)
    Optimized for pre-computed w2v-BERT features
    """
    
    def __init__(
        self,
        finetune_config_path: str = "finetune_precomputed_config.yaml",
        **kwargs
    ):
        super().__init__()

        # Load config
        self.cfg = OmegaConf.load(finetune_config_path)
        self.model_dir = self.cfg['model_dir']
        
        # Training parameters
        self.learning_rate = self.cfg['training']['learning_rate']
        self.warmup_steps = self.cfg['training']['warmup_steps']
        self.max_steps = self.cfg['training']['max_steps']
        self.gradient_clip_val = self.cfg['training']['gradient_clip_val']
        self.freeze_conditioning = self.cfg['freeze_conditioning']

        # Save hyperparameters
        self.save_hyperparameters()

        # Initialize GPT-only models and auxiliary components
        self._init_models()

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def print_all_modules(self):
        """Print all modules in the model"""
        print("\n" + "="*60)
        print("ALL MODULES IN IndexTTSLightningModulePrecomputed")
        print("="*60)
        for name, module in self.named_modules():
            print(f"{name}: {type(module).__name__}")
        print("="*60 + "\n")
        
    def _init_models(self):
        """Initialize GPT-only models for finetuning"""
        print(">> Initializing GPT-only mode for finetuning (PRECOMPUTED FEATURES)")
        
        # Load full UnifiedVoice to extract GPT components
        full_model = UnifiedVoice(**self.cfg.gpt)
        gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(full_model, gpt_path)
        print(f">> Full model weights loaded from: {gpt_path}")

        # Extract GPT components from the loaded model
        self.gpt = full_model.gpt  # The main GPT transformer
        self.mel_pos_embedding = full_model.mel_pos_embedding
        self.text_pos_embedding = full_model.text_pos_embedding
        self.mel_layer_pos_embedding = getattr(full_model, 'mel_layer_pos_embedding', None)
        self.text_layer_pos_embedding = getattr(full_model, 'text_layer_pos_embedding', None)
        
        # Extract embeddings and heads
        self.text_embedding = full_model.text_embedding
        self.mel_embedding = full_model.mel_embedding
        self.final_norm = full_model.final_norm
        self.text_head = full_model.text_head
        self.mel_head = full_model.mel_head

        # Initialize and store conditioning encoders for inference (but freeze them)
        self.cond_num = 32 # same as Index TTS 2
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True) # same as Index TTS 2
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True) # same as Index TTS 2
        self.conditioning_encoder = full_model.conditioning_encoder
        self.emo_conditioning_encoder = full_model.emo_conditioning_encoder
        self.perceiver_encoder = getattr(full_model, 'perceiver_encoder', None)
        self.emo_perceiver_encoder = getattr(full_model, 'emo_perceiver_encoder', None)
        
        # Store other necessary components
        self.emovec_layer = full_model.emovec_layer
        self.emo_layer = full_model.emo_layer
        self.speed_emb = full_model.speed_emb
        
        # Keep reference to config parameters
        self.number_text_tokens = full_model.number_text_tokens
        self.number_mel_codes = full_model.number_mel_codes
        self.start_text_token = full_model.start_text_token
        self.stop_text_token = full_model.stop_text_token
        self.start_mel_token = full_model.start_mel_token
        self.stop_mel_token = full_model.stop_mel_token
        self.max_mel_tokens = full_model.max_mel_tokens
        self.max_text_tokens = full_model.max_text_tokens
        self.model_dim = full_model.model_dim
        self.condition_type = full_model.condition_type # conformer_perceiver
        print(">> GPT components extracted successfully")

        # For precomputed features, we still need semantic codec but NOT the feature extractor
        # Initialize semantic codec for mel code generation
        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec
        self.semantic_codec.eval()
        print(f'>> semantic_codec weights restored from: {semantic_code_ckpt}')
        
        # Clean up full model to save memory
        del full_model
        
        # Freeze models that we don't want to train
        if self.freeze_conditioning:
            self._freeze_conditioning_components()
            
        print("✅ PRECOMPUTED FEATURES MODE: Skipping SeamlessM4TFeatureExtractor and semantic_model initialization")
            
    def setup(self, stage: str = None):
        """Called at the beginning of fit, validate, test, or predict"""
        print(f"===== SETUP CALLED (PRECOMPUTED) ===== Stage: {stage}")
        
        # Move semantic codec to correct device
        self.semantic_codec = self.semantic_codec.to(self.device)
        
        # Log model info to wandb if available (only on fit stage)
        if stage == "fit" and hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            self._log_model_info()
            
    def _freeze_conditioning_components(self):
        """Freeze conditioning-related components"""
        # Freeze semantic codec
        for param in self.semantic_codec.parameters():
            param.requires_grad = False
            
        # Freeze conditioning encoders (we only train the GPT transformer)
        if hasattr(self, 'conditioning_encoder'):
            for param in self.conditioning_encoder.parameters():
                param.requires_grad = False
        if hasattr(self, 'emo_conditioning_encoder'):
            for param in self.emo_conditioning_encoder.parameters():
                param.requires_grad = False
        if hasattr(self, 'perceiver_encoder') and self.perceiver_encoder is not None:
            for param in self.perceiver_encoder.parameters():
                param.requires_grad = False
        if hasattr(self, 'emo_perceiver_encoder') and self.emo_perceiver_encoder is not None:
            for param in self.emo_perceiver_encoder.parameters():
                param.requires_grad = False
        
        # Freeze emotion processing layers
        if hasattr(self, 'emovec_layer'):
            for param in self.emovec_layer.parameters():
                param.requires_grad = False
        if hasattr(self, 'emo_layer'):
            for param in self.emo_layer.parameters():
                param.requires_grad = False
        
        # Freeze speed embeddings
        if hasattr(self, 'speed_emb'):
            for param in self.speed_emb.parameters():
                param.requires_grad = False
                
    def _log_model_info(self):
        """Log model architecture information to wandb"""
        try:
            import wandb
            
            # Count parameters for each component
            gpt_params = sum(p.numel() for p in self.gpt.parameters())
            text_emb_params = sum(p.numel() for p in self.text_embedding.parameters())
            mel_emb_params = sum(p.numel() for p in self.mel_embedding.parameters())
            text_head_params = sum(p.numel() for p in self.text_head.parameters())
            mel_head_params = sum(p.numel() for p in self.mel_head.parameters())
            
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            
            # Log model architecture summary
            self.logger.experiment.log({
                "model_info/gpt_params": gpt_params,
                "model_info/text_embedding_params": text_emb_params,
                "model_info/mel_embedding_params": mel_emb_params,
                "model_info/text_head_params": text_head_params,
                "model_info/mel_head_params": mel_head_params,
                "model_info/total_trainable_params": trainable_params,
                "model_info/total_params": total_params,
                "model_info/trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
                "model_info/model_dim": self.model_dim,
                "model_info/max_text_tokens": self.max_text_tokens,
                "model_info/max_mel_tokens": self.max_mel_tokens,
                "model_info/freeze_conditioning": self.freeze_conditioning,
                "model_info/precomputed_features": True,
            })
            
        except ImportError:
            print("Warning: wandb not available, skipping model info logging")
        except Exception as e:
            print(f"Warning: Failed to log model info to wandb: {e}")
        
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Only optimize GPT transformer and related components
        trainable_params = []
        
        # Add GPT transformer parameters
        trainable_params.extend(self.gpt.parameters())
        
        # Add embedding and head parameters
        trainable_params.extend(self.text_embedding.parameters())
        trainable_params.extend(self.mel_embedding.parameters())
        trainable_params.extend(self.final_norm.parameters())
        trainable_params.extend(self.text_head.parameters())
        trainable_params.extend(self.mel_head.parameters())
        
        # Add positional embeddings
        trainable_params.extend(self.mel_pos_embedding.parameters())
        trainable_params.extend(self.text_pos_embedding.parameters())
        if self.mel_layer_pos_embedding is not None:
            trainable_params.extend(self.mel_layer_pos_embedding.parameters())
        if self.text_layer_pos_embedding is not None:
            trainable_params.extend(self.text_layer_pos_embedding.parameters())
        
        # Filter for parameters that require gradients
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        # Get optimizer config
        opt_cfg = self.cfg['training']['optimizer']
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=opt_cfg['betas'],
            weight_decay=opt_cfg['weight_decay'],
            eps=opt_cfg['eps']
        )
        
        # Use ReduceLROnPlateau scheduler for validation-based learning rate reduction
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # minimize validation loss
            factor=0.5,  # reduce LR by factor of 0.5
            patience=5,  # number of epochs with no improvement after which LR will be reduced
            min_lr=1e-6  # minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # ReduceLROnPlateau works per epoch
                "frequency": 1,
                "monitor": "val_loss",  # Monitor validation loss
            }
        }

    def get_speech_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2), cond_mel_lengths)  # (b, s, d), (b, 1, s) -> (B, 160, 512), (B, 1, 160)
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, num_latents, d) -> (B, 32, 1280)
        return conds

    def get_emo_conditioning(self, emo_conditioning_input, emo_cond_mel_lengths=None):
        emo_conditioning_input, mask = self.emo_conditioning_encoder(emo_conditioning_input.transpose(1, 2),
                                                                        emo_cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(emo_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens
    
    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar
        
    def generate_mel_codes(self, conditioning_latents: torch.Tensor) -> torch.Tensor:
        """
        Generate mel codes from conditioning latents using semantic codec
        Args:
            conditioning_latents: [batch, time, 1024]
        Returns:
            mel_codes: [batch, time]
        """
        with torch.no_grad():
            # Quantize to get mel codes
            _, mel_codes = self.semantic_codec.quantize(conditioning_latents)
            mel_codes = mel_codes.squeeze(1)  # Remove quantizer dimension
        return mel_codes
    
    def get_logits(self, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False, return_latent=False):
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]

        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with PRE-COMPUTED features - optimized for speed"""
        
        # ===== PRECOMPUTED FEATURES: Direct usage =====
        # Use pre-computed semantic embeddings directly (no feature extraction needed)
        spk_cond_emb = batch["semantic_emb"].to(self.device)  # [batch, time, 1024]
        spk_cond_lengths = batch["semantic_lengths"].to(self.device)  # [batch]
        
        # Speech conditioning (same as original)
        speech_conditioning_latent = self.get_speech_conditioning(spk_cond_emb.transpose(1,2), spk_cond_lengths)

        # Emotion conditioning (reuse same embeddings)
        emo_cond_emb = spk_cond_emb  # Same pre-computed embeddings
        emo_cond_lengths = spk_cond_lengths  # Same lengths
        emo_vec_syn_ori = self.get_emo_conditioning(emo_cond_emb.transpose(1,2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)

        # Set up text & mel codes with proper padding (same as original)
        text_inputs = batch["text_tokens"].to(self.device)
        text_lengths = batch["text_lengths"].to(self.device)
        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        # Generate mel codes from semantic codec (same as original)
        mel_codes, rec_feat = self.semantic_codec.quantize(spk_cond_emb)
        mel_codes_lengths = spk_cond_lengths
        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        # Duration embedding (same as original)
        use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        conds = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)

        # Build inputs and targets (same as original)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
        mel_emb = self.mel_embedding(mel_codes)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)
        
        # Get logits (same as original)
        text_logits, mel_logits = self.get_logits(speech_conditioning_inputs=conds, 
                                                  first_inputs=text_emb, 
                                                  first_head=self.text_head, 
                                                  second_inputs=mel_emb, 
                                                  second_head=self.mel_head, 
                                                  get_attns=False, 
                                                  return_latent=False)

        # Loss calculation (same as original)
        text_logits = text_logits.permute(0, 2, 1)  # [batch, seq_len, vocab_size]
        mel_logits = mel_logits.permute(0, 2, 1)    # [batch, seq_len, vocab_size]
        
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_targets.reshape(-1).long()
        )
        
        mel_loss = F.cross_entropy(
            mel_logits.reshape(-1, mel_logits.size(-1)),
            mel_targets.reshape(-1).long()
        )        

        # Combine losses (same as original)
        total_loss = 0.1 * text_loss + mel_loss
        print(f"===== PRECOMPUTED DEBUG =====: total_loss: {total_loss.item()}, text_loss: {text_loss.item()}, mel_loss: {mel_loss.item()}")
        
        # Log metrics
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("text_loss", text_loss, on_step=True, on_epoch=True)
        self.log("mel_loss", mel_loss, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True)

        return total_loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with PRE-COMPUTED features"""
        
        with torch.no_grad():
            # ===== PRECOMPUTED FEATURES: Direct usage =====
            spk_cond_emb = batch["semantic_emb"].to(self.device)
            spk_cond_lengths = batch["semantic_lengths"].to(self.device)
            
            # Speech conditioning
            speech_conditioning_latent = self.get_speech_conditioning(spk_cond_emb.transpose(1,2), spk_cond_lengths)

            # Emotion conditioning
            emo_cond_emb = spk_cond_emb
            emo_cond_lengths = spk_cond_lengths
            emo_vec_syn_ori = self.get_emo_conditioning(emo_cond_emb.transpose(1,2), emo_cond_lengths)
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
            emo_vec = self.emo_layer(emo_vec_syn)

            # Set up text & mel codes
            text_inputs = batch["text_tokens"].to(self.device)
            text_lengths = batch["text_lengths"].to(self.device)
            text_inputs = self.set_text_padding(text_inputs, text_lengths)
            text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

            # Generate mel codes
            mel_codes, rec_feat = self.semantic_codec.quantize(spk_cond_emb)
            mel_codes_lengths = spk_cond_lengths
            mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
            mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

            # Duration embedding
            use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
            duration_emb = self.speed_emb(torch.zeros_like(use_speed))
            duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
            conds = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)

            # Build inputs and targets
            text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
            text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
            mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
            mel_emb = self.mel_embedding(mel_codes)
            mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)
            
            # Get logits
            text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=False, return_latent=False)
            
            # Loss calculation
            text_logits = text_logits.permute(0, 2, 1)
            mel_logits = mel_logits.permute(0, 2, 1)
            
            text_loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                text_targets.reshape(-1).long()
            )
            
            mel_loss = F.cross_entropy(
                mel_logits.reshape(-1, mel_logits.size(-1)),
                mel_targets.reshape(-1).long()
            )
            
            total_loss = 0.1 * text_loss + mel_loss
        
        # Log validation metrics
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_text_loss", text_loss, on_step=False, on_epoch=True)
        self.log("val_mel_loss", mel_loss, on_step=False, on_epoch=True)
        
        return total_loss
        
    def configure_gradient_clipping(
        self, 
        optimizer, 
        gradient_clip_val, 
        gradient_clip_algorithm
    ):
        """Configure gradient clipping"""
        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=gradient_clip_val, 
                gradient_clip_algorithm=gradient_clip_algorithm
            )
            
    def on_save_checkpoint(self, checkpoint):
        """Save GPT model components"""
        # Save GPT components separately
        checkpoint["gpt_state_dict"] = self.gpt.state_dict()
        checkpoint["text_embedding_state_dict"] = self.text_embedding.state_dict()
        checkpoint["mel_embedding_state_dict"] = self.mel_embedding.state_dict()
        checkpoint["final_norm_state_dict"] = self.final_norm.state_dict()
        checkpoint["text_head_state_dict"] = self.text_head.state_dict()
        checkpoint["mel_head_state_dict"] = self.mel_head.state_dict()
        checkpoint["mel_pos_embedding_state_dict"] = self.mel_pos_embedding.state_dict()
        checkpoint["text_pos_embedding_state_dict"] = self.text_pos_embedding.state_dict()
        
        # Save layer pos embeddings if they exist
        if self.mel_layer_pos_embedding is not None:
            checkpoint["mel_layer_pos_embedding_state_dict"] = self.mel_layer_pos_embedding.state_dict()
        if self.text_layer_pos_embedding is not None:
            checkpoint["text_layer_pos_embedding_state_dict"] = self.text_layer_pos_embedding.state_dict()
        return checkpoint
        
    def on_load_checkpoint(self, checkpoint):
        """Load GPT model components"""
        # Load GPT components separately
        if "gpt_state_dict" in checkpoint:
            self.gpt.load_state_dict(checkpoint["gpt_state_dict"])
        if "text_embedding_state_dict" in checkpoint:
            self.text_embedding.load_state_dict(checkpoint["text_embedding_state_dict"])
        if "mel_embedding_state_dict" in checkpoint:
            self.mel_embedding.load_state_dict(checkpoint["mel_embedding_state_dict"])
        if "final_norm_state_dict" in checkpoint:
            self.final_norm.load_state_dict(checkpoint["final_norm_state_dict"])
        if "text_head_state_dict" in checkpoint:
            self.text_head.load_state_dict(checkpoint["text_head_state_dict"])
        if "mel_head_state_dict" in checkpoint:
            self.mel_head.load_state_dict(checkpoint["mel_head_state_dict"])
        if "mel_pos_embedding_state_dict" in checkpoint:
            self.mel_pos_embedding.load_state_dict(checkpoint["mel_pos_embedding_state_dict"])
        if "text_pos_embedding_state_dict" in checkpoint:
            self.text_pos_embedding.load_state_dict(checkpoint["text_pos_embedding_state_dict"])
        
        # Load layer pos embeddings if they exist
        if "mel_layer_pos_embedding_state_dict" in checkpoint and self.mel_layer_pos_embedding is not None:
            self.mel_layer_pos_embedding.load_state_dict(checkpoint["mel_layer_pos_embedding_state_dict"])
        if "text_layer_pos_embedding_state_dict" in checkpoint and self.text_layer_pos_embedding is not None:
            self.text_layer_pos_embedding.load_state_dict(checkpoint["text_layer_pos_embedding_state_dict"])
    
    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer step to handle DDP synchronization issues"""
        
        # Additional gradient clipping for DDP stability
        if self.trainer.world_size > 1:
            # Synchronize gradients manually if needed
            torch.distributed.barrier()
    
    def configure_model(self):
        """Configure model for DDP to avoid autograd hooks issues"""
        # This is called before DDP wrapping
        # Ensure all parameters are properly initialized
        for name, param in self.named_parameters():
            if param.requires_grad and not hasattr(param, 'grad'):
                param.grad = None


def test():
    # Test the lightning module
    print("Testing IndexTTS Lightning Module (Precomputed)...")
    
    module = IndexTTSLightningModulePrecomputed(
        finetune_config_path="/data/user_data/willw2/expressive_s2st/index-tts/config/finetune_precomputed_config.yaml"
    )
    
    print("Lightning module initialized successfully!")
    print("PRECOMPUTED FEATURES mode enabled")


if __name__ == "__main__":
    test()