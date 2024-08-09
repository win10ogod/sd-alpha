import argparse
import logging
import os
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, get_scheduler
from diffusers import AutoencoderKL, PNDMScheduler
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Text-to-Image Diffusion Transformer")
    parser.add_argument("--dataset", type=str, default="E:/aesthetic")
    parser.add_argument("--output_dir", type=str, default="./diffusion_transformer_checkpoints")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--preprocess", action="store_true", help="Preprocess images and save as npy files")
    return parser.parse_args()

class ImageTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_npy=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_npy = use_npy
        self.supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff')
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(self.supported_extensions)]
        self.text_cache = {}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        txt_path = os.path.join(self.root_dir, os.path.splitext(img_name)[0] + '.txt')

        if self.use_npy:
            npy_path = os.path.join(self.root_dir, "npy", os.path.splitext(img_name)[0] + '.npy')
            try:
                latents = np.load(npy_path)
            except IOError:
                print(f"Error opening npy file: {npy_path}")
                return None
        else:
            try:
                image = Image.open(img_path).convert('RGB')  # Convert to RGB
            except IOError:
                print(f"Error opening image file: {img_path}")
                return None

            if self.transform:
                image = self.transform(image)

        if txt_path not in self.text_cache:
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                self.text_cache[txt_path] = text
            except IOError:
                print(f"Error opening text file: {txt_path}")
                return None
        else:
            text = self.text_cache[txt_path]

        if self.use_npy:
            return {'latents': latents, 'text': text}
        else:
            return {'image': image, 'text': text}

def collate_fn(examples):
    if 'latents' in examples[0]:
        latents = torch.tensor(np.stack([example['latents'] for example in examples]))
    else:
        images = torch.stack([example['image'] for example in examples])
    texts = [example['text'] for example in examples]
    return {'latents': latents if 'latents' in locals() else None, 'pixel_values': images if 'images' in locals() else None, 'text': texts}

class unet(UNet2DConditionModel):
    def __init__(self, image_size, hidden_size):
        super().__init__(
            sample_size=image_size // 8,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=[
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ],
            up_block_types=[
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ],
            cross_attention_dim=hidden_size,
            attention_head_dim=8,
            norm_num_groups=16,
            norm_eps=1e-05,
            act_fn="silu",
            downsample_padding=1,
            flip_sin_to_cos=True,
            freq_shift=0,
            only_cross_attention= False,
            use_linear_projection= True,
            upcast_attention= True,
            center_input_sample=False,
            mid_block_scale_factor=1,
        )

def convert_images_to_npy(root_dir, output_dir, image_size, vae, device):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff')
    image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(supported_extensions)]

    os.makedirs(output_dir, exist_ok=True)

    for img_name in image_files:
        img_path = os.path.join(root_dir, img_name)
        npy_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')

        if os.path.exists(npy_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')  # Convert to RGB
        except IOError:
            print(f"Error opening image file: {img_path}")
            continue

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        np.save(npy_path, latents.cpu().numpy())

def train():
    args = parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    noise_scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.bos_token
    text_encoder = AutoModel.from_pretrained("Qwen/Qwen2-1.5B-Instruct", trust_remote_code=True).to(device)

    feature_extractor = CLIPImageProcessor.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="feature_extractor")

    model = unet(args.image_size, text_encoder.config.hidden_size).to(device)

    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if args.preprocess:
        root_dir = args.dataset
        output_dir = os.path.join(args.dataset, "npy")
        convert_images_to_npy(root_dir, output_dir, args.image_size, vae, device)

    dataset = ImageTextDataset(root_dir=args.dataset, transform=transform, use_npy=args.preprocess)

    if args.local_rank != -1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    if args.mixed_precision != "no":
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(args.num_train_epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            with autocast(enabled=args.mixed_precision != "no"):
                if batch['latents'] is not None:
                    latents = batch['latents'].to(device)
                else:
                    pixel_values = batch["pixel_values"].to(device)
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                # Ensure latents are in the correct shape
                if latents.dim() == 5:
                    latents = latents.squeeze(1)  # Remove the singleton dimension

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                text_inputs = tokenizer(batch["text"], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
                text_input_ids = text_inputs.input_ids.to(device)

                with torch.no_grad():
                    text_outputs = text_encoder(
                                    input_ids=text_input_ids,
                                    attention_mask=text_inputs.attention_mask.to(device)
                                                )
                text_embeddings = text_outputs.last_hidden_state

                noise_pred = model(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                lr_scheduler.step()

            progress_bar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})

            if (step + 1) % args.save_steps == 0 and args.local_rank in [-1, 0]:
                pipeline = StableDiffusionPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=model,
                    scheduler=noise_scheduler,
                    safety_checker=None,
                    feature_extractor=feature_extractor,
                )
                pipeline.save_pretrained(args.output_dir)

        if args.local_rank in [-1, 0]:
            pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=model,
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
            )
            pipeline.save_pretrained(args.output_dir)

if __name__ == "__main__":
    train()