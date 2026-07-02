"""
CALM Stage-1: pretrain a per-entity modality encoder (imaging E_I or genetics E_G).

Select the modality with --modality {imaging,genetics}. Both use the same
PathwayEncoder architecture (per-ROI for imaging, per-pathway for genetics);
this script trains it on ASD/CON classification and saves a per-fold checkpoint
consumed by the Stage-2 alignment run via
    main.py --pretrained_{imaging,genetics}_checkpoint <abs path>.

Paper (Sec. 2.4): modality encoders + classifiers are pretrained for 50 epochs.

Examples
--------
    # smoke test (synthetic, no data/paths needed):
    python3 train_stage1.py --modality imaging  --use_synthetic_data --pretrain_epochs=2 --normalization=layer
    python3 train_stage1.py --modality genetics --use_synthetic_data --pretrain_epochs=2 --normalization=layer --genetics_input_channels=6

    # real run, one fold:
    python3 train_stage1.py --modality imaging --test_fold=0 --pretrain_epochs=50 \
        --output_features=6 --normalization=layer --batch_size=64 \
        --learning_rate=1e-3 --imaging_feature_noise=0.15 \
        --stage1_output_dir=/abs/path/to/checkpoints/imaging

    python3 train_stage1.py --modality genetics --test_fold=0 --pretrain_epochs=50 \
        --output_features=6 --normalization=layer --batch_size=64 \
        --genetics_input_channels=6 --genetics_learning_rate=1e-3 \
        --stage1_output_dir=/abs/path/to/checkpoints/genetics
"""
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from utils.add_argument import add_argument
from utils.utils import seed_everything

# Reuse the exact loaders + encoder builders + forwards used by the alignment stage.
from main import (
    load_synthetic_data,
    load_imaging_data,
    load_genetics_data,
    load_pretrained_imaging_encoder,
    load_pretrained_genetics_encoder_classifier,
    forward_imaging_encoder,
    forward_genetics_encoder,
)


def build_modality(hparams, device):
    """Resolve everything that differs between the two Stage-1 encoders.

    Returns train_loader, val_loader, encoder, lr, batch_key,
    run_forward(encoder, x, training) -> logits, and save_extra (modality-specific
    checkpoint fields).
    """
    synthetic = getattr(hparams, "use_synthetic_data", False)

    if hparams.modality == "imaging":
        if synthetic:
            (train_loader, val_loader, _), _, n_entities, _ = load_synthetic_data(hparams)
        else:
            train_loader, val_loader, _, n_entities = load_imaging_data(hparams)
        # No --pretrained_imaging_checkpoint -> builds a fresh encoder.
        encoder, _ = load_pretrained_imaging_encoder(hparams, n_entities, device)
        lr = hparams.learning_rate
        batch_key = "img_feat"

        def run_forward(enc, x, training):
            noise = hparams.imaging_feature_noise if training else 0.0
            _, logits = forward_imaging_encoder(enc, x, noise, training=training)
            return logits

        save_extra = {"n_rois": n_entities}
    else:  # genetics
        if synthetic:
            _, (train_loader, val_loader, _), _, n_entities = load_synthetic_data(hparams)
        else:
            train_loader, val_loader, _, n_entities = load_genetics_data(hparams)
        # No --pretrained_genetics_checkpoint -> builds a fresh encoder.
        encoder = load_pretrained_genetics_encoder_classifier(hparams, n_entities, device)
        lr = hparams.genetics_learning_rate
        batch_key = "pathway"

        def run_forward(enc, x, training):
            _, logits = forward_genetics_encoder(enc, x, return_reconstruction=False)
            return logits

        save_extra = {"input_channels": hparams.genetics_input_channels, "n_pathway": n_entities}

    return train_loader, val_loader, encoder, lr, batch_key, run_forward, save_extra


def evaluate(encoder, loader, device, batch_key, run_forward):
    encoder.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[batch_key].to(device).float()
            logits = run_forward(encoder, x, training=False)
            probs.append(torch.sigmoid(logits.squeeze(-1)).cpu())
            labels.append(batch["label"].view(-1))
    p = torch.cat(probs).numpy()
    l = torch.cat(labels).numpy()
    try:
        return accuracy_score(l, (p > 0.5).astype(int)), roc_auc_score(l, p)
    except Exception:
        return 0.0, 0.0


def main(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (train_loader, val_loader, encoder, lr, batch_key,
     run_forward, save_extra) = build_modality(hparams, device)
    encoder.train()

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=lr,
        weight_decay=hparams.genetics_weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nStage-1 {hparams.modality} pretraining: {hparams.pretrain_epochs} epochs, "
          f"fold {hparams.test_fold}\n")
    for epoch in range(hparams.pretrain_epochs):
        encoder.train()
        last_loss = 0.0
        for batch in train_loader:
            x = batch[batch_key].to(device).float()
            labels = batch["label"].to(device).view(-1).float()
            logits = run_forward(encoder, x, training=True)
            loss = criterion(logits.squeeze(-1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        val_acc, val_auc = evaluate(encoder, val_loader, device, batch_key, run_forward)
        print(
            f"[fold {hparams.test_fold}] epoch {epoch+1}/{hparams.pretrain_epochs} "
            f"loss {last_loss:.4f}  val acc/auc {val_acc:.3f}/{val_auc:.3f}",
            flush=True,
        )

    out_dir = Path(hparams.stage1_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"stage1_{hparams.modality}_fold{hparams.test_fold}.pth"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "output_features": hparams.output_features,
            "epoch": hparams.pretrain_epochs,
            **save_extra,
        },
        ckpt,
    )
    print(f"\n✓ saved {hparams.modality} Stage-1 checkpoint: {ckpt}")


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    add_argument(parser)
    parser.add_argument(
        "--modality",
        choices=["imaging", "genetics"],
        required=True,
        help="Which Stage-1 encoder to pretrain (imaging E_I or genetics E_G).",
    )
    hparams = parser.parse_args()
    seed_everything(42)
    main(hparams)
