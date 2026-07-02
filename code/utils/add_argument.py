from argparse import ArgumentParser


def add_argument(parser: ArgumentParser):
    parser.add_argument(
        "--tensor_board_logger",
        default=r"/projectnb/ace-genetics/jueqiw/experiment/CrossModalityLearning/tensorboard/",
        help="TensorBoardLogger dir",
    )
    parser.add_argument(
        "--experiment_name",
        default="attention",
        help="Experiment name for TensorBoardLogger",
    )
    parser.add_argument("--test_fold", default=0, type=int)
    parser.add_argument("--run_time", default=0, type=int)
    parser.add_argument("--dataset", choices=["ACE", "ADNI", "SSC"], default="ACE")
    parser.add_argument(
        "--genetics_input_channels",
        default=1,
        type=int,
        help="Number of input channels for pathway encoder (e.g., 1 for single GWAS, 2 for dual GWAS inputs).",
    )
    parser.add_argument("--classifier_latent_dim", default=64, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--n_epochs", default=3000, type=int)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--normalization",
        choices=["batch", "layer", "instance", "None"],
        default="instance",
        type=str,
    )
    parser.add_argument(
        "--hidden_dim_qk",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--hidden_dim_k",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--hidden_dim_v",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--soft_sign_constant",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--alignment_coral_weight",
        default=1.0,
        type=float,
        help="Weight for class-conditional CORAL loss during alignment stage",
    )
    parser.add_argument(
        "--alignment_contrastive_weight",
        default=1.0,
        type=float,
        help="Weight for class-level contrastive loss during alignment stage",
    )
    parser.add_argument(
        "--alignment_orthogonality_weight",
        default=1e-3,
        type=float,
        help="Weight for orthogonality regularizer on projection matrices",
    )
    parser.add_argument(
        "--alignment_tau",
        default=0.07,
        type=float,
        help="Temperature parameter for alignment contrastive loss",
    )
    parser.add_argument(
        "--alignment_lambda",
        default=0.5,
        type=float,
        help="Balance parameter for bidirectional contrastive loss",
    )
    parser.add_argument(
        "--imaging_cls_weight",
        default=1.0,
        type=float,
        help="Weight for imaging classification loss during alignment",
    )
    parser.add_argument(
        "--genetics_cls_weight",
        default=1.0,
        type=float,
        help="Weight for genetics classification loss during alignment",
    )
    # CALM paper (Sec. 2.4) two-stage / imaging-encoder knobs
    parser.add_argument(
        "--encoder_finetune_lr",
        default=1e-5,
        type=float,
        help="Learning rate for fine-tuning the imaging encoder during alignment (paper: 1e-5).",
    )
    parser.add_argument(
        "--pretrain_epochs",
        default=50,
        type=int,
        help="Stage 1: epochs to pretrain modality encoders + classifiers (paper: 50).",
    )
    parser.add_argument(
        "--align_epochs",
        default=30,
        type=int,
        help="Stage 2: epochs to train the linear projections (paper: 30).",
    )
    parser.add_argument(
        "--imaging_feature_noise",
        default=0.15,
        type=float,
        help="Std of Gaussian noise added to imaging ROI features during training (paper: 0.15).",
    )
    parser.add_argument(
        "--use_synthetic_data",
        action="store_true",
        help="Run on randomly-generated pseudo data (no real data/SCC paths needed). "
        "Use for a quick end-to-end smoke test that the pipeline is runnable.",
    )
    parser.add_argument(
        "--stage1_output_dir",
        default="/path/to/checkpoints",
        type=str,
        help="Directory where Stage-1 encoder checkpoints are saved (consumed by "
        "main.py --pretrained_{imaging,genetics}_checkpoint). CHANGE to your path.",
    )
    parser.add_argument(
        "--genetics_learning_rate",
        default=5e-4,
        type=float,
        help="Learning rate for genetics encoder and alignment projections",
    )
    parser.add_argument(
        "--genetics_weight_decay",
        default=1e-4,
        type=float,
        help="Weight decay for genetics encoder and alignment projections",
    )
    parser.add_argument(
        "--relu_at_coattention",
        action="store_true",
    )
    parser.add_argument(
        "--hidden_dim_q",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--encoder_hidden_dim_1",
        default=32,
        type=int,
        help="First encoder layer dimension for genetics pathway encoder (replaces hidden_dim_q in genetics model)",
    )
    parser.add_argument(
        "--encoder_hidden_dim_2",
        default=16,
        type=int,
        help="Second encoder layer dimension for genetics pathway encoder (replaces hidden_dim_qk in genetics model)",
    )
    parser.add_argument(
        "--encoder_hidden_dim_3",
        default=None,
        type=int,
        help="Optional third encoder layer dimension for genetics pathway encoder (set to use a 3-layer MLP)",
    )
    parser.add_argument(
        "--align_projector_only",
        action="store_true",
        help="Freeze imaging/genetics encoders and classifiers; train only the shared projector with alignment losses.",
    )
    parser.add_argument(
        "--freeze_genetics_layers",
        default=0,
        type=int,
        help="Freeze the first N genetics encoder layers (1=encoder_linear_1, 2=+encoder_linear_2, 3=+encoder_linear_3, 4=+encoder_linear_out).",
    )
    parser.add_argument(
        "--use_mmd_alignment",
        action="store_true",
        help="Use MMD loss instead of CORAL for modality alignment.",
    )
    parser.add_argument(
        "--encoder_dims",
        default=None,
        type=str,
        help="Comma-separated list of encoder hidden dimensions (excluding output_features). "
        "E.g., '1,6' for 1→6→output_features encoder. "
        "If provided, overrides encoder_hidden_dim_1 and encoder_hidden_dim_2. "
        "Decoder will automatically mirror: output_features→6→1",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
        help="Default weight decay (deprecated, use specific ones below)",
    )
    parser.add_argument(
        "--classifier_weight_decay",
        default=0.05,
        type=float,
        help="Weight decay for classifier and enhanced layers",
    )
    parser.add_argument(
        "--encoder_weight_decay",
        default=0.01,
        type=float,
        help="Weight decay for transformer encoder layers",
    )
    parser.add_argument(
        # would also start the whole run five folder cross validation
        "--not_write_tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "--n_folder",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--classifier_drop_out",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision (FP16) training for memory efficiency and speed",
    )
    # Loss function parameters
    # Learning rate scheduler parameters
    parser.add_argument(
        "--gamma",
        default=0.95,
        type=float,
        help="Gamma parameter for scheduler (decay rate for exponential/step, factor for plateau)",
    )
    # Model architecture parameters
    # Gradual unfreezing parameters
    parser.add_argument(
        "--output_features",
        default=256,
        type=int,
        help="Number of output feature channels from enhanced encoder (8, 16, 32, 64, 128, 256). Lower values reduce classifier parameters for small datasets.",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable multi-GPU training using DataParallel",
    )
    # Layer-wise Learning Rate Decay (LLRD) parameters
    parser.add_argument(
        "--use_llrd",
        action="store_true",
        help="Use layer-wise learning rate decay instead of uniform learning rate",
    )
    parser.add_argument(
        "--llrd_decay_factor",
        default=0.8,
        type=float,
        help="Layer-wise learning rate decay factor (default: 0.8)",
    )
    parser.add_argument(
        "--unfreeze_last_n_blocks",
        default=3,
        type=int,
        help="Number of last transformer blocks to unfreeze (2-4, default: 3)",
    )
    parser.add_argument(
        "--classifier_lr",
        default=1e-3,
        type=float,
        help="Learning rate for classifier when using LLRD (default: 1e-3)",
    )
    parser.add_argument(
        "--encoder_base_lr",
        default=1e-5,
        type=float,
        help="Base learning rate for encoder layers when using LLRD (default: 1e-5)",
    )
    parser.add_argument(
        "--cosine_schedule",
        action="store_true",
        help="Use cosine learning rate schedule with warmup",
    )
    parser.add_argument(
        "--warmup_steps",
        default=150,
        type=int,
        help="Number of warmup steps for cosine schedule (default: 150, ~2-3 epochs for typical batch sizes)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Enable test mode: use only 10% of training data for faster debugging",
    )
    parser.add_argument(
        "--test_mode_ratio",
        default=0.1,
        type=float,
        help="Ratio of data to use in test mode (default: 0.1 = 10%)",
    )
    parser.add_argument(
        "--use_reconstruction",
        action="store_true",
        help="Use reconstruction regularization for FreeSurfer features [4, 210]",
    )
    parser.add_argument(
        "--reconstruction_weight",
        type=float,
        default=0.05,
        help="Weight for reconstruction loss (auxiliary task for generalization)",
    )
    parser.add_argument(
        "--pathway_dropout",
        type=float,
        default=0.0,
        help="Dropout rate for input pathways (0.0-0.3). Randomly drops entire pathways during training for robustness.",
    )
    parser.add_argument(
        "--encoder_dropout",
        type=float,
        default=0.0,
        help="Dropout rate for encoder hidden layers (0.0-0.2). Regularizes pathway transformations.",
    )
    parser.add_argument(
        "--decoder_lr",
        type=float,
        default=None,
        help=(
            "Learning rate for reconstruction decoder (default: use classifier_lr). "
            "Set higher (e.g., 2e-3) for faster reconstruction learning."
        ),
    )
    parser.add_argument(
        "--freeze_encoder_decoder",
        action="store_true",
        help="Stage 2: Freeze encoder and decoder weights, train only classifier.",
    )
    parser.add_argument(
        "--freeze_pretrained",
        action="store_true",
        help="Freeze all pretrained base encoder (UNeST) layers. Enhanced layers (encoder0/1/2) and decoder remain trainable.",
    )
    parser.add_argument(
        "--freeze_encoder_block5",
        action="store_true",
        help="Freeze transformer_encoder.5 in level 2 (only unfreeze blocks 6-7 instead of 5-6-7)",
    )
    parser.add_argument(
        "--freeze_encoder_block6",
        action="store_true",
        help="Freeze transformer_encoder.6 in level 2 (only unfreeze block 7)",
    )
    parser.add_argument(
        "--unfreeze_level0",
        action="store_true",
        help="Unfreeze Level 0's last transformer block (transformer_encoder.1) with very low LR. Use for reconstruction learning. Keeps patch_embed and transformer_encoder.0 frozen.",
    )
    parser.add_argument(
        "--block7_use_classifier_lr",
        action="store_true",
        help="Use classifier LR and weight decay for transformer_encoder.7 (treat it as task-specific layer)",
    )
    parser.add_argument(
        "--block7_lr",
        default=None,
        type=float,
        help="Custom learning rate for transformer_encoder.7 in level 2 (overrides --block7_use_classifier_lr if set)",
    )
    parser.add_argument(
        "--block6_lr",
        default=None,
        type=float,
        help="Custom learning rate for transformer_encoder.6 in level 2",
    )

    # OneCycleLR scheduler parameters
    parser.add_argument(
        "--use_onecycle",
        action="store_true",
        help="Use OneCycleLR scheduler instead of cosine annealing (automatically sets max_lr per parameter group)",
    )
    parser.add_argument(
        "--onecycle_pct_start",
        default=0.1,
        type=float,
        help="Percentage of cycle spent warming up (default: 0.1 = 10%)",
    )
    parser.add_argument(
        "--onecycle_div_factor",
        default=25,
        type=float,
        help="Initial LR = max_lr / div_factor (default: 25)",
    )
    parser.add_argument(
        "--onecycle_final_div_factor",
        default=1e4,
        type=float,
        help="Final LR = max_lr / (div_factor * final_div_factor) (default: 1e4)",
    )
    # Pretrained model checkpoint paths for alignment stage
    parser.add_argument(
        "--pretrained_imaging_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained imaging encoder checkpoint (.pth file)",
    )
    parser.add_argument(
        "--pretrained_genetics_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained genetics encoder checkpoint (.pth file)",
    )
