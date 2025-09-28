import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

class VLMDataset_mmvet(Dataset):
    def __init__(self, json_path, image_base_dir, processor, device="cuda"):
        self.device = device
        self.processor = processor
        self.image_base_dir = image_base_dir  # Base directory where images are stored
        self.data = []  # Store (image_path, safe_instruction, embedding, category) tuples

        # Load JSON file
        with open(json_path, "r") as f:
            json_data = json.load(f)

        # Process JSON data
        for item in json_data:
            item_id = item["id"]

            # Construct correct image path
            image_path = os.path.join(self.image_base_dir, item["image_path"].split("images/")[1])

            # Check if image exists
            if os.path.exists(image_path):
                safe_instruction = item["question"]
                
                # Assign category = 0 for all safe data
                category = 0
                
                # Generate a random embedding vector of size 4096
                embedding = torch.tensor(np.random.rand(4096), dtype=torch.float32)
                
                self.data.append((image_path, safe_instruction, embedding, category))
            else:
                print(f"Warning: Image {image_path} not found, skipping.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, safe_instruction, embedding, category = self.data[idx]

        # Load and preprocess image
        image = Image.open(image_path)
        # Convert palette images with transparency to RGBA first, then to RGB
        if image.mode in ('P', 'LA'):
            image = image.convert('RGBA')
        image = image.convert("RGB").resize((224, 224))

        return image, safe_instruction, embedding, torch.tensor(category, dtype=torch.int64)

# Custom `collate_fn` for dynamic padding and adding `safe=1`
def collate_fn(batch):
    images, texts, embeddings, categories = zip(*batch)  # Unzip batch into image, text, embedding, and category tuples

    # Process images and text with CLIP processor
    #inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding="longest")
    inputs = processor(
        text=list(texts), 
        images=list(images), 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=77  # Fixes mismatch
    )

    # Add safe=1 label
    inputs["safe"] = torch.ones(len(batch), dtype=torch.int64).to(device)  # Batch size safe labels

    # Add embeddings
    inputs["embedding"] = torch.stack(embeddings).to(device)
    
    # Add category (always 0 for safe dataset)
    inputs["category"] = torch.tensor(categories, dtype=torch.int64).to(device)

    return {k: v.to(device) for k, v in inputs.items()}  # Move tensors to device


def main(model,processor):
    # Load CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define correct paths
    image_base_dir = "./data/mm-vet/images"
    json_path = "./data/mm-vet/sample.json"
    # Create dataset and dataloader
    safe_dataset_mmvet = VLMDataset_mmvet(json_path, image_base_dir, processor, device)
    safe_dataloader_mmvet = DataLoader(safe_dataset_mmvet, batch_size=8, shuffle=True, collate_fn=collate_fn)

    return safe_dataset_mmvet, safe_dataloader_mmvet
