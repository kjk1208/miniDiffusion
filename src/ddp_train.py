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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from model.vae import VAE, load_vae
from model.dit_components import HandlePrompt
from torch.optim.lr_scheduler import LambdaLR
from model.clip import CLIP, load_clip, OpenCLIP
from model.t5_encoder import T5EncoderModel, load_t5
from common_ds import get_dataset, get_fashion_mnist_dataset
import torch.distributed as dist

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

def train(args):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    is_main = rank == 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("model", timestamp) if is_main else None
    checkpoint_dir = os.path.join(log_dir, "checkpoints") if is_main else None
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)

    device = f"cuda:{rank}"
    model = DiT().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(2025)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    scheduler = LambdaLR(optimizer, lambda step: step / 100.0 if step < 100 else 1.0)

    vae = load_vae(model=VAE(), device=device)
    clip = load_clip(model=CLIP(), model_2=OpenCLIP(), device=device)
    t5_encoder = load_t5(model=T5EncoderModel(), device=device)
    prompt_handler = HandlePrompt()

    if not args.train_dataset:
        train_dataset, val_loader = get_fashion_mnist_dataset(batch_size=args.batch_size, device=device)
    else:
        train_dataset = get_dataset(args.train_dataset, batch_size=args.batch_size, device=device)
        val_loader = get_dataset(args.val_dataset, batch_size=args.batch_size, device=device) if args.val_dataset else None

    train_sampler = DistributedSampler(train_dataset.dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset.dataset, batch_size=1, sampler=train_sampler, num_workers=0, collate_fn=lambda x: x
    )

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Epoch {epoch + 1}/{args.epochs}]", dynamic_ncols=True) if is_main else enumerate(train_loader)

        for i, batch in pbar:
            model.train()
            latent, noised_image, added_noise, timesteps, label = stack_tensors(batch)
            embeddings, pooled_embeddings = prompt_handler(label, clip=clip, clip_2=clip.model_2, t5_encoder=t5_encoder)
            target = added_noise - noised_image

            drift = model(latent=latent, timestep=timesteps,
                          encoder_hidden_states=embeddings,
                          pooled_projections=pooled_embeddings)
            loss = loss_fn(drift, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            for layer in model.module.model:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.reset_cache()

            total_loss += loss.item()
            if is_main and args.enable_log:
                log_loss_csv(os.path.join(log_dir, "train_log.csv"), epoch + 1, i + 1, loss.item())
                train_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())

        if is_main:
            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch + 1}] Train Loss: {avg_loss:.4f}")

            if (epoch + 1) % 5 == 0:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, f"model_epoch{epoch + 1}.pth"))

            if val_loader:
                val_loss = evaluate(model.module, val_loader, clip, clip.model_2, t5_encoder, prompt_handler, device)
                log_loss_csv(os.path.join(log_dir, "val_log.csv"), epoch + 1, 0, val_loss)
                val_losses.append(val_loss)
                print(f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "best_checkpoint.pth"))

    if is_main:
        torch.save(model.module.state_dict(), os.path.join(log_dir, "final_checkpoint.pth"))
        if args.enable_log:
            save_loss_plot(train_losses, os.path.join(log_dir, "train_loss_curve.png"), "Train")
            if val_losses:
                save_loss_plot(val_losses, os.path.join(log_dir, "val_loss_curve.png"), "Val")

    dist.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--val_dataset", type=str, default=None)
    parser.add_argument("--enable_log", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
