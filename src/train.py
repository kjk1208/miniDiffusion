import os
import csv
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from model.dit import DiT
from common import loss_fn
from typing import Optional
from model.vae import VAE, load_vae
from model.dit_components import HandlePrompt
from torch.optim.lr_scheduler import LambdaLR
from model.clip import CLIP, load_clip, OpenCLIP
from model.t5_encoder import T5EncoderModel, load_t5
from common_ds import get_dataset, get_fashion_mnist_dataset


def log_loss_csv(log_path, epoch, step, loss_val):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "loss"])
        if write_header:
            writer.writeheader()
        writer.writerow({"epoch": epoch, "step": step, "loss": loss_val})


def save_loss_plot(loss_list, path, label):
    plt.figure()
    plt.plot(loss_list, label=label)
    plt.xlabel("Epoch" if "val" in label.lower() else "Iteration")
    plt.ylabel("Loss")
    plt.title(f"{label} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def stack_tensors(batch):
    latent = torch.stack([sample[0] for sample in batch]).squeeze(1)
    noised_image = torch.stack([sample[1] for sample in batch]).squeeze(1)
    added_noise = torch.stack([sample[2] for sample in batch]).squeeze(1)
    timesteps = torch.stack([sample[3] for sample in batch]).squeeze(1)
    label = torch.stack([sample[4] for sample in batch]).squeeze(1)
    return latent, noised_image, added_noise, timesteps, label


def check_dataset_dir(path: str):
    assert os.path.exists(path), f"Dataset path '{path}' does not exist."
    return True


def evaluate(model, dataloader, clip, clip_2, t5_encoder, prompt_handler, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            latent, noised_image, added_noise, timesteps, label = stack_tensors(batch)
            embeddings, pooled_embeddings = prompt_handler(label, clip=clip, clip_2=clip_2, t5_encoder=t5_encoder)
            target = added_noise - noised_image

            drift = model(latent=latent,
                          timestep=timesteps,
                          encoder_hidden_states=embeddings,
                          pooled_projections=pooled_embeddings)

            loss = loss_fn(drift, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(device: Optional[str], train_dataset: Optional[str], val_dataset: Optional[str],
          epochs: int = 3, lr: float = 0.0001, batch_size: int = 8, log: bool = True):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("model", timestamp)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    train_log_path = os.path.join(log_dir, "train_log.csv")
    val_log_path = os.path.join(log_dir, "val_log.csv")
    final_ckpt_path = os.path.join(log_dir, "final_checkpoint.pth")
    train_plot_path = os.path.join(log_dir, "train_loss_curve.png")
    val_plot_path = os.path.join(log_dir, "val_loss_curve.png")

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = DiT().to(device=device)

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.manual_seed(2025)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  fused=True if torch.cuda.is_available() else False)

    warmup_steps = 100
    def lr_lambda(current_step):
        return float(current_step) / float(max(1, warmup_steps)) if current_step < warmup_steps else 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    vae = load_vae(model=VAE(), device=device)
    clip = load_clip(model=CLIP(), model_2=OpenCLIP(), device=device)
    t5_encoder = load_t5(model=T5EncoderModel(), device=device)
    prompt_handler = HandlePrompt()

    # Dataset loading
    if not train_dataset:
        print("Defaulting to Fashion MNIST")
        train_dataset, val_loader = get_fashion_mnist_dataset(batch_size=batch_size, device=device)
    else:
        check_dataset_dir(train_dataset)
        train_dataset = get_dataset(train_dataset, batch_size=batch_size, device=device)

        val_loader = None
        if val_dataset:
            check_dataset_dir(val_dataset)
            val_loader = get_dataset(val_dataset, batch_size=batch_size, device=device)
        else:
            val_loader = None

    print(f"[{timestamp}] Starting to train...")
    model.train()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        total = train_dataset.dataset.max_samples // batch_size
        pbar = tqdm(enumerate(train_dataset), total=total, desc=f"[Epoch {epoch + 1}/{epochs}]", dynamic_ncols=True)

        for i, batch in pbar:
            model.train()
            latent, noised_image, added_noise, timesteps, label = stack_tensors(batch)
            embeddings, pooled_embeddings = prompt_handler(label, clip=clip, clip_2=clip.model_2, t5_encoder=t5_encoder)
            target = added_noise - noised_image

            drift = model(latent=latent, timestep=timesteps,
                          encoder_hidden_states=embeddings,
                          pooled_projections=pooled_embeddings)

            loss = loss_fn(drift, target)

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            if log:
                train_losses.append(loss.item())
                log_loss_csv(train_log_path, epoch + 1, i + 1, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            for layer in model.model:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.reset_cache()

            pbar.set_postfix(loss=loss.item())

        print(f"[Epoch {epoch + 1}] Train Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch{epoch + 1}.pth"))

        # Validation
        if val_loader:
            val_loss = evaluate(model, val_loader, clip, clip.model_2, t5_encoder, prompt_handler, device)
            val_losses.append(val_loss)
            print(f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f}")
            if log:
                log_loss_csv(val_log_path, epoch + 1, 0, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model to {best_path} (val_loss = {val_loss:.4f})")

    torch.save(model.state_dict(), final_ckpt_path)

    if log:
        save_loss_plot(train_losses, train_plot_path, label="Train")
        if val_loader:
            save_loss_plot(val_losses, val_plot_path, label="Val")


def get_args():
    parser = argparse.ArgumentParser(description="Train a model with train and validation logs.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--train_dataset", type=str, required=False)
    parser.add_argument("--val_dataset", type=str, required=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--log", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(
        device=args.device,
        train_dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        log=args.log
    )
