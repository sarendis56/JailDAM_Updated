import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import random

class VLMDataset_medical_vqa(Dataset):
    def __init__(self, processor, device="cuda", max_samples=1000, split="train"):
        """
        Medical VQA dataset loader from Hugging Face
        
        Args:
            processor: CLIP processor for text and image processing
            device: Device to use (cuda/cpu)
            max_samples: Maximum number of samples to load (for balanced testing)
            split: Dataset split to use ("train")
        """
        self.device = device
        self.processor = processor
        self.data = []  # Store (image_path, question_text, embedding, category) tuples
        
        print(f"Loading VQA-RAD dataset from Hugging Face...")
        print(f"Split: {split}, Max samples: {max_samples}")
        
        # Load VQA-RAD dataset from Hugging Face with images
        try:
            dataset = load_dataset("flaviagiammarino/vqa-rad", split=split)
            print(f"Successfully loaded VQA-RAD {split} set with {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading VQA-RAD dataset: {e}")
            print("Trying alternative medical VQA dataset...")
            try:
                # Try another medical VQA dataset
                dataset = load_dataset("medarc/vqa-rad", split=split)
                print(f"Successfully loaded VQA-RAD (medarc) {split} set with {len(dataset)} samples")
            except Exception as e2:
                print(f"Error loading alternative dataset: {e2}")
                print("Falling back to a smaller subset...")
                # Try to load a smaller subset if full dataset fails
                dataset = load_dataset("flaviagiammarino/vqa-rad", split=split, streaming=True)
                dataset = list(dataset.take(max_samples))
        
        # Convert dataset to list if it's not already
        if hasattr(dataset, '__iter__') and not isinstance(dataset, list):
            dataset = list(dataset)
        
        # Limit the number of samples for balanced testing
        if len(dataset) > max_samples:
            # Randomly sample to get balanced data
            dataset = random.sample(dataset, max_samples)
            print(f"Randomly sampled {max_samples} samples from VQA-RAD")
        
        # Process the dataset
        print(f"Dataset structure sample: {list(dataset[0].keys()) if len(dataset) > 0 else 'Empty dataset'}")
        if len(dataset) > 0:
            print(f"Image type: {type(dataset[0]['image'])}")
            print(f"Question type: {type(dataset[0].get('question', 'No question field'))}")
        
        for i, item in enumerate(dataset):
            try:
                # Extract image and question from VQA-RAD structure
                image_data = item['image']
                
                # Handle different image formats
                if isinstance(image_data, str):
                    # If it's a file path, try to load it
                    try:
                        image = Image.open(image_data)
                    except Exception as e:
                        print(f"Warning: Could not load image from {image_data}: {e}, skipping...")
                        continue
                elif hasattr(image_data, 'mode'):
                    # If it's already a PIL Image
                    image = image_data
                else:
                    # Try to convert other formats to PIL Image
                    try:
                        image = Image.fromarray(image_data)
                    except Exception as e:
                        print(f"Warning: Could not convert image data: {e}, skipping...")
                        continue
                
                # Extract question from VQA-RAD structure
                question = item.get('question', '')
                if not question:
                    # Try alternative question fields
                    question = item.get('question_text', item.get('q', ''))
                
                if not question:
                    print(f"Warning: No question found in sample {i}, skipping...")
                    continue
                
                # Clean up question
                question = question.replace('<image>', '').replace('\n', ' ').strip()
                
                # Convert PIL image to RGB if needed
                if hasattr(image, 'mode') and image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Generate a random embedding vector of size 4096 (same as other datasets)
                embedding = torch.tensor(np.random.rand(4096), dtype=torch.float32)
                
                # Assign category = 0 for all safe data (Medical VQA is benign)
                category = 0
                
                self.data.append((image, question, embedding, category))
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} samples...")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        print(f"VQA-RAD dataset loaded successfully with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, question_text, embedding, category = self.data[idx]

        # Resize image to standard size
        image = image.resize((224, 224))

        return image, question_text, embedding, torch.tensor(category, dtype=torch.int64)

# Custom `collate_fn` for dynamic padding and adding `safe=1`
def collate_fn(batch):
    images, question_texts, embeddings, categories = zip(*batch)  # Unzip batch into image, text, embedding, and category tuples
    
    # Create processor instance
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Process images and text with CLIP processor
    inputs = processor(
        text=list(question_texts), 
        images=list(images), 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=77  # Fixes mismatch
    )

    # Add safe=1 label (VQA-RAD is benign)
    inputs["safe"] = torch.ones(len(batch), dtype=torch.int64).to(device)  # Batch size safe labels

    # Add embeddings
    inputs["embedding"] = torch.stack(embeddings).to(device)
    
    # Add category (always 0 for safe dataset)
    inputs["category"] = torch.tensor(categories, dtype=torch.int64).to(device)
    
    # Add texts for compatibility
    inputs["texts"] = list(question_texts)
    inputs["images"] = list(images)

    # Move only tensors to device, keep lists as they are
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    
    return inputs


def main(model, processor, max_samples=1000, split="train"):
    """
    Main function to create VQA-RAD dataset and dataloader
    
    Args:
        model: CLIP model (not used but kept for compatibility)
        processor: CLIP processor
        max_samples: Maximum number of samples to load
        split: Dataset split to use ("train")
    """
    # Load CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dataset and dataloader
    medical_vqa_dataset = VLMDataset_medical_vqa(processor, device, max_samples, split)
    medical_vqa_dataloader = DataLoader(medical_vqa_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    return medical_vqa_dataset, medical_vqa_dataloader
