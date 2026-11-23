"""ECS 174 HW2 training for the CelebA DCGAN.

tutorial: the tutorial generator and discriminator, one update each per batch.
d5: the discriminator every batch, the generator every fifth batch.
simple: the two-convolution discriminator from the assignment.
"""

import argparse
import json
import os
import random
import time

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

torch.set_num_threads(16)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATAROOT = os.path.join(ROOT, "data", "celeba")
RUNS = os.path.join(ROOT, "runs")

batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 0
manual_seed = 999


class Generator(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class SimpleDiscriminator(nn.Module):
    """Two convolutions, then a 1x1 average pool and a sigmoid."""

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_grid(images, path):
    grid = vutils.make_grid(images, padding=2, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_losses(d_losses, g_losses, g_steps, path):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_steps, g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train_one(mode, epochs, max_images, workers):
    out = os.path.join(RUNS, mode)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "log.txt")
    ckpt_path = os.path.join(out, "checkpoint.pt")

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    dataset = dset.ImageFolder(
        root=DATAROOT,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    if max_images and max_images < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(max_images)))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    device = torch.device("cpu")

    d_repeat = 5 if mode == "d5" else 1
    netG = Generator(ngpu).to(device)
    if mode == "simple":
        netD = SimpleDiscriminator(ngpu).to(device)
    else:
        netD = Discriminator(ngpu).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    with open(os.path.join(out, "models.txt"), "w", encoding="utf-8") as f:
        f.write(str(netG) + "\n\n" + str(netD) + "\n")

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    G_losses = []
    D_losses = []
    g_steps = []
    start_epoch = 0
    iters = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("finished"):
            plot_losses(
                ckpt["D_losses"],
                ckpt["G_losses"],
                ckpt["g_steps"],
                os.path.join(out, "loss_curve.png"),
            )
            print(f"{mode} already finished")
            return
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        optimizerG.load_state_dict(ckpt["optG"])
        optimizerD.load_state_dict(ckpt["optD"])
        fixed_noise = ckpt["fixed_noise"]
        G_losses = ckpt["G_losses"]
        D_losses = ckpt["D_losses"]
        g_steps = ckpt["g_steps"]
        start_epoch = ckpt["epoch"] + 1
        iters = ckpt["iters"]
        print(f"Resuming {mode} at epoch {start_epoch}")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if start_epoch == 0 and iters == 0:
        real_batch = next(iter(dataloader))
        save_grid(real_batch[0][:64], os.path.join(out, "training_images.png"))
        note = f"dataset images: {len(dataset)}"
        if max_images and max_images < 202599:
            note += f" (first {len(dataset)} of 202599 CelebA images)"
        log(note)
        log(f"dataloader workers: {workers}")
        log(f"batches per epoch: {len(dataloader)}")
        log("Starting Training Loop...")

    for epoch in range(start_epoch, epochs):
        epoch_t = time.perf_counter()
        last_errG = float("nan")
        last_dg2 = float("nan")
        for i, data in enumerate(dataloader):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            if iters % d_repeat == 0:
                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                last_dg2 = output.mean().item()
                last_errG = errG.item()
                optimizerG.step()
                G_losses.append(last_errG)
                g_steps.append(iters)

            D_losses.append(errD.item())

            if i % 50 == 0:
                log(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        last_errG,
                        D_x,
                        D_G_z1,
                        last_dg2,
                    )
                )

            if iters % 500 == 0 or (
                epoch == epochs - 1 and i == len(dataloader) - 1
            ):
                with torch.no_grad():
                    snap = netG(fixed_noise).detach().cpu()
                save_grid(snap, os.path.join(out, f"snap_{iters:06d}.png"))

            iters += 1

        with torch.no_grad():
            epoch_fake = netG(fixed_noise).detach().cpu()
        save_grid(epoch_fake, os.path.join(out, f"epoch_{epoch + 1}.png"))
        elapsed = time.perf_counter() - epoch_t
        log(f"epoch {epoch + 1} done in {elapsed:.1f}s")
        torch.save(
            {
                "epoch": epoch,
                "iters": iters,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optG": optimizerG.state_dict(),
                "optD": optimizerD.state_dict(),
                "fixed_noise": fixed_noise,
                "G_losses": G_losses,
                "D_losses": D_losses,
                "g_steps": g_steps,
                "finished": epoch + 1 >= epochs,
            },
            ckpt_path,
        )

    plot_losses(D_losses, G_losses, g_steps, os.path.join(out, "loss_curve.png"))
    with torch.no_grad():
        final_fake = netG(fixed_noise).detach().cpu()
    real_batch = next(iter(dataloader))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.axis("off")
    ax.set_title("Real Images")
    ax.imshow(
        np.transpose(
            vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).numpy(),
            (1, 2, 0),
        )
    )
    ax = fig.add_subplot(1, 2, 2)
    ax.axis("off")
    ax.set_title("Fake Images")
    ax.imshow(
        np.transpose(
            vutils.make_grid(final_fake, padding=5, normalize=True).numpy(),
            (1, 2, 0),
        )
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out, "real_vs_fake.png"), dpi=120)
    plt.close()

    summary = {
        "mode": mode,
        "epochs": epochs,
        "d_updates": len(D_losses),
        "g_updates": len(G_losses),
        "final_loss_d": D_losses[-1] if D_losses else None,
        "final_loss_g": G_losses[-1] if G_losses else None,
        "mean_loss_d_last_200": float(np.mean(D_losses[-200:])) if D_losses else None,
        "mean_loss_g_last_50": float(np.mean(G_losses[-50:])) if G_losses else None,
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log("finished " + json.dumps(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all", choices=["all", "tutorial", "d5", "simple"])
    parser.add_argument("--epochs", type=int, default=num_epochs)
    parser.add_argument("--max-images", type=int, default=12800)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    modes = ["tutorial", "d5", "simple"] if args.only == "all" else [args.only]
    for mode in modes:
        train_one(mode, args.epochs, args.max_images, args.workers)


if __name__ == "__main__":
    main()
