import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from datasets import load_dataset


class VLMDatasetAlpaca(Dataset):
    def __init__(self, processor, device="cuda", max_samples=500, dataset_name="HuggingFaceH4/CodeAlpaca_20K", split="train"):
        self.device = device
        self.processor = processor
        self.data = []  # (dummy_image, instruction_text, embedding, category=0)

        # Try to load flexible splits and fallbacks
        ds = None
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception:
            try:
                # Load without split and pick an available one
                ds_all = load_dataset(dataset_name)
                for candidate in ["train", "validation", "test", "train[:%d]" % max_samples]:
                    if isinstance(ds_all, dict) and candidate in ds_all:
                        ds = ds_all[candidate]
                        break
                if ds is None:
                    # Pick the first available split
                    if isinstance(ds_all, dict) and len(ds_all) > 0:
                        ds = next(iter(ds_all.values()))
            except Exception:
                # Last resort: try streaming small take
                try:
                    ds_stream = load_dataset(dataset_name, split=split, streaming=True)
                    ds = list(ds_stream.take(max_samples))
                except Exception:
                    ds = []

        if hasattr(ds, "__iter__") and not isinstance(ds, list):
            ds = list(ds)

        if len(ds) > max_samples:
            ds = ds[:max_samples]

        # Prepare a single dummy RGB image used for all samples
        dummy_image = Image.new("RGB", (224, 224), color=(127, 127, 127))

        for i, item in enumerate(ds):
            # Probe common text/instruction fields across varied datasets
            instr = (
                item.get("instruction")
                or item.get("instructions")
                or item.get("prompt")
                or item.get("input")
                or item.get("question")
                or item.get("title")
                or item.get("abstract")
                or item.get("body")
                or item.get("utterance")
                or item.get("text")
            )
            if not instr:
                continue
            instr = str(instr).replace("<image>", "").strip()

            embedding = torch.tensor(np.random.rand(4096), dtype=torch.float32)
            category = 0  # benign
            self.data.append((dummy_image.copy(), instr, embedding, category))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, text, embedding, category = self.data[idx]
        return image, text, embedding, torch.tensor(category, dtype=torch.int64)


def collate_fn(batch):
    images, texts, embeddings, categories = zip(*batch)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=77,
    )
    inputs["safe"] = torch.ones(len(batch), dtype=torch.int64).to(device)
    inputs["embedding"] = torch.stack(embeddings).to(device)
    inputs["category"] = torch.tensor(categories, dtype=torch.int64).to(device)
    inputs["texts"] = list(texts)
    inputs["images"] = list(images)

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


def main(model, processor, max_samples=500, split="train"):
    dataset = VLMDatasetAlpaca(processor, max_samples=max_samples, split=split)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    return dataset, dataloader
