import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine Annealing scheduler with linear warmup

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR (default: 0.0)
        last_epoch: Last epoch index (default: -1)
    """

    def __init__(
        self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0  # Track actual training steps
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        """Override step to track training steps, not epochs"""
        self.current_step += 1
        super().step(epoch)

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup: 0 → target_lr over warmup_steps
            return [
                base_lr * self.current_step / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing: target_lr → min_lr over remaining steps
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            # Clamp progress to [0, 1] to avoid negative values
            progress = max(0.0, min(1.0, progress))
            return [
                self.min_lr_ratio * base_lr
                + (base_lr - self.min_lr_ratio * base_lr)
                * 0.5
                * (1.0 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def create_optimizer_and_scheduler(model, hparams, total_steps):
    """
    Create optimizer and scheduler based on hyperparameters

    Args:
        model: The model to optimize
        hparams: Hyperparameters namespace
        total_steps: Total number of training steps

    Returns:
        optimizer: Configured optimizer
        scheduler: Configured scheduler (None if not using cosine schedule)
    """

    if hparams.use_llrd:
        print("Using Layer-wise Learning Rate Decay (LLRD)")

        # Apply gradual unfreezing strategy
        if hasattr(model, "apply_gradual_unfreezing_strategy"):
            model.apply_gradual_unfreezing_strategy(hparams.unfreeze_last_n_blocks)

        # Get parameter groups with different learning rates
        if hasattr(model, "get_layerwise_lr_param_groups"):
            param_groups = model.get_layerwise_lr_param_groups(
                classifier_lr=hparams.classifier_lr,
                encoder_base_lr=hparams.encoder_base_lr,
                decay_factor=hparams.llrd_decay_factor,
            )
        else:
            # Fallback for models without LLRD support
            print("Model doesn't support LLRD, falling back to uniform LR")
            param_groups = [{"params": model.parameters(), "lr": hparams.learning_rate}]
    else:
        print("Using uniform learning rate across all parameters")
        param_groups = [{"params": model.parameters(), "lr": hparams.learning_rate}]

    # Create optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=hparams.weight_decay if hasattr(hparams, "weight_decay") else 0.05,
    )

    print(f"Optimizer created with {len(param_groups)} parameter groups")
    for i, group in enumerate(param_groups):
        group_name = group.get("name", f"group_{i}")
        print(
            f"  {group_name}: LR = {group['lr']:.2e}, params = {len(group['params'])}"
        )

    # Create scheduler if requested
    scheduler = None
    if hparams.cosine_schedule:
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=hparams.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.0,
        )
        print(f"Cosine scheduler created with {hparams.warmup_steps} warmup steps")

    return optimizer, scheduler


def setup_training_components(encoder_model, classifier_model, hparams, total_steps):
    """
    Setup all training components with LLRD support

    Args:
        encoder_model: The image encoder model
        classifier_model: The classifier model
        hparams: Hyperparameters
        total_steps: Total training steps

    Returns:
        optimizer: Combined optimizer
        scheduler: Learning rate scheduler (if enabled)
    """

    if hparams.use_llrd:
        print("Setting up LLRD for encoder + classifier")

        # Get the actual model (handle DataParallel wrapper)
        actual_encoder = (
            encoder_model.module if hasattr(encoder_model, "module") else encoder_model
        )

        # Apply gradual unfreezing to encoder - BUT skip if freeze_pretrained is set
        freeze_pretrained = (
            hparams.freeze_pretrained
            if hasattr(hparams, "freeze_pretrained")
            else False
        )
        if freeze_pretrained:
            print("Skipping gradual unfreezing - freeze_pretrained is enabled")
        elif hasattr(actual_encoder, "apply_gradual_unfreezing_strategy"):
            print("Applying gradual unfreezing strategy...")
            freeze_block5 = (
                hparams.freeze_encoder_block5
                if hasattr(hparams, "freeze_encoder_block5")
                else False
            )
            freeze_block6 = (
                hparams.freeze_encoder_block6
                if hasattr(hparams, "freeze_encoder_block6")
                else False
            )
            unfreeze_level0 = (
                hparams.unfreeze_level0
                if hasattr(hparams, "unfreeze_level0")
                else False
            )
            actual_encoder.apply_gradual_unfreezing_strategy(
                hparams.unfreeze_last_n_blocks,
                freeze_encoder_block5=freeze_block5,
                freeze_encoder_block6=freeze_block6,
                unfreeze_level0=unfreeze_level0
            )
        else:
            print(
                "Warning: Model doesn't have apply_gradual_unfreezing_strategy method"
            )

        # Get encoder parameter groups
        encoder_param_groups = []
        if hasattr(actual_encoder, "get_layerwise_lr_param_groups"):
            print(
                f"DEBUG: Setting enhanced layers classifier_lr to {hparams.classifier_lr:.2e}"
            )
            block7_classifier_lr = (
                hparams.block7_use_classifier_lr
                if hasattr(hparams, "block7_use_classifier_lr")
                else False
            )
            block7_custom_lr = (
                hparams.block7_lr
                if hasattr(hparams, "block7_lr")
                else None
            )
            block6_custom_lr = (
                hparams.block6_lr
                if hasattr(hparams, "block6_lr")
                else None
            )
            decoder_custom_lr = (
                hparams.decoder_lr
                if hasattr(hparams, "decoder_lr")
                else None
            )
            encoder_param_groups = actual_encoder.get_layerwise_lr_param_groups(
                classifier_lr=hparams.classifier_lr,  # Enhanced layers should match classifier LR
                encoder_base_lr=hparams.encoder_base_lr,
                decay_factor=hparams.llrd_decay_factor,
                block7_use_classifier_lr=block7_classifier_lr,
                block7_lr=block7_custom_lr,
                block6_lr=block6_custom_lr,
                decoder_lr=decoder_custom_lr,
                classifier_weight_decay=(
                    hparams.classifier_weight_decay
                    if hasattr(hparams, "classifier_weight_decay")
                    else 0.05
                ),
                encoder_weight_decay=(
                    hparams.encoder_weight_decay
                    if hasattr(hparams, "encoder_weight_decay")
                    else 0.01
                ),
            )
        else:
            print("Warning: Model doesn't have get_layerwise_lr_param_groups method")

        # Get classifier parameters (always get the head learning rate)
        # Separate bias/norm params from weight params for classifier
        classifier_weights = []
        classifier_bias_norm = []

        for name, param in classifier_model.named_parameters():
            if param.requires_grad:
                # Shape-based rule: 1D parameters (norms/bias) get no weight decay
                if len(param.shape) == 1 or name.endswith(".bias"):
                    classifier_bias_norm.append(param)
                else:
                    classifier_weights.append(param)

        classifier_groups = []
        # Add classifier weight parameters with weight decay
        if classifier_weights:
            classifier_groups.append(
                {
                    "params": classifier_weights,
                    "lr": hparams.classifier_lr,
                    "weight_decay": (
                        hparams.classifier_weight_decay
                        if hasattr(hparams, "classifier_weight_decay")
                        else 0.05
                    ),
                    "name": "classifier_weights",
                }
            )

        # Add classifier bias/norm parameters without weight decay
        if classifier_bias_norm:
            classifier_groups.append(
                {
                    "params": classifier_bias_norm,
                    "lr": hparams.classifier_lr,
                    "weight_decay": 0.0,
                    "name": "classifier_bias_norm",
                }
            )

        # Combine parameter groups
        all_param_groups = encoder_param_groups + classifier_groups

    else:
        print("Using uniform learning rate for encoder + classifier")

        # Get the actual model (handle DataParallel wrapper)
        actual_encoder = (
            encoder_model.module if hasattr(encoder_model, "module") else encoder_model
        )

        # Apply gradual unfreezing to encoder (same strategy as LLRD) - BUT skip if freeze_pretrained is set
        freeze_pretrained = (
            hparams.freeze_pretrained
            if hasattr(hparams, "freeze_pretrained")
            else False
        )
        if freeze_pretrained:
            print("Skipping gradual unfreezing - freeze_pretrained is enabled")
        elif hasattr(actual_encoder, "apply_gradual_unfreezing_strategy"):
            print("Applying gradual unfreezing strategy for uniform LR...")
            unfreeze_level0 = (
                hparams.unfreeze_level0
                if hasattr(hparams, "unfreeze_level0")
                else False
            )
            actual_encoder.apply_gradual_unfreezing_strategy(
                hparams.unfreeze_last_n_blocks,
                unfreeze_level0=unfreeze_level0
            )
        else:
            print(
                "Warning: Model doesn't have apply_gradual_unfreezing_strategy method"
            )

        # Get all trainable parameters (after freezing is applied)
        encoder_params = [p for p in encoder_model.parameters() if p.requires_grad]
        classifier_params = [
            p for p in classifier_model.parameters() if p.requires_grad
        ]
        all_params = encoder_params + classifier_params

        all_param_groups = [
            {"params": all_params, "lr": hparams.learning_rate, "name": "all"}
        ]

    # Create optimizer
    optimizer = torch.optim.AdamW(
        all_param_groups,
        weight_decay=hparams.weight_decay if hasattr(hparams, "weight_decay") else 0.05,
    )

    print(f"\nOptimizer summary:")
    print(f"  Total parameter groups: {len(all_param_groups)}")
    for group in all_param_groups:
        group_name = group.get("name", "unnamed")
        num_params = sum(p.numel() for p in group["params"])
        print(f"    {group_name}: LR = {group['lr']:.2e}, parameters = {num_params:,}")

    # Create scheduler if requested
    scheduler = None
    if hasattr(hparams, "use_onecycle") and hparams.use_onecycle:
        # OneCycleLR scheduler with per-group max_lr
        # Extract max_lr from each parameter group
        max_lrs = [group["lr"] for group in all_param_groups]

        # Use total_steps directly (since scheduler.step() is called per batch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,  # Total training steps (batches)
            pct_start=(
                hparams.onecycle_pct_start
                if hasattr(hparams, "onecycle_pct_start")
                else 0.1
            ),
            anneal_strategy="cos",
            div_factor=(
                hparams.onecycle_div_factor
                if hasattr(hparams, "onecycle_div_factor")
                else 25
            ),
            final_div_factor=(
                hparams.onecycle_final_div_factor
                if hasattr(hparams, "onecycle_final_div_factor")
                else 1e4
            ),
            three_phase=False,
        )
        print(
            f"  OneCycleLR scheduler: max_lr={max_lrs}, total_steps={total_steps}, "
            f"pct_start={hparams.onecycle_pct_start if hasattr(hparams, 'onecycle_pct_start') else 0.1}, "
            f"div_factor={hparams.onecycle_div_factor if hasattr(hparams, 'onecycle_div_factor') else 25}"
        )
    elif hparams.cosine_schedule:
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=hparams.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.0,
        )
        print(
            f"  Cosine scheduler: {hparams.warmup_steps} warmup steps, {total_steps} total steps"
        )

    return optimizer, scheduler
