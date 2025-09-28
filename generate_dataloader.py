from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.cuda.amp import autocast
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import json
import gc
import random
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# Load CLIP processor (which contains the tokenizer)

class ConceptDataset(Dataset):
    def __init__(self, text_sims, image_sims, labels):
        self.text_sims = text_sims  # (total_samples, 1300, 1)
        self.image_sims = image_sims  # (total_samples, 1300, 1)
        self.labels = labels  # (total_samples,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_sims[idx], self.image_sims[idx], self.labels[idx]
    
def decode_clip_input_ids(input_ids):
    """
    Converts CLIP tokenized input_ids back into human-readable text.
    
    Args:
        input_ids (torch.Tensor): Tensor of tokenized IDs (batch_size, seq_len).
    
    Returns:
        List[str]: Decoded text prompts.
    """
    decoded_texts = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return decoded_texts
def compute_concept_similarities(dataloader, model, concept_embeddings,processor, device="cuda"):
    all_text_sims, all_image_sims, all_decoded_texts = [], [], []

    # Ensure concept embeddings are on the correct device
    concept_embeddings = concept_embeddings.to(device)
    
    total_batches = len(dataloader)
    print(f"Computing concept similarities for {total_batches} batches...")
    print("-" * 50)

    for batch_idx, batch in enumerate(dataloader):
        torch.cuda.empty_cache()  # Free up memory
        gc.collect()  # Run garbage collection

        with torch.no_grad():
            with autocast():  # Enable FP16 precision
                # Move batch tensors to device
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Decode input_ids back to text
                decoded_texts = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                all_decoded_texts.extend(decoded_texts)  # Store decoded texts

                # Get embeddings
                image_embeddings = model.get_image_features(pixel_values=pixel_values)
                text_embeddings = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

                # Normalize embeddings (L2 normalization for cosine similarity)
                image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
                text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)

                # Get CLIP's logit scale (temperature scaling)
                logit_scale = model.logit_scale.exp()

                # Compute cosine similarities (before softmax)
                text_sim = torch.matmul(text_embeddings, concept_embeddings.T) * logit_scale  # (batch, 1300)
                image_sim = torch.matmul(image_embeddings, concept_embeddings.T) * logit_scale  # (batch, 1300)

                # Apply softmax across **concepts** (last dimension) for proper normalization
                text_sim = torch.nn.functional.softmax(text_sim, dim=-1).unsqueeze(-1)  # (batch, 1300, 1)
                image_sim = torch.nn.functional.softmax(image_sim, dim=-1).unsqueeze(-1)  # (batch, 1300, 1)

                # Append batch results
                all_text_sims.append(text_sim)
                all_image_sims.append(image_sim)
        
        # Progress reporting
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            progress = ((batch_idx + 1) / total_batches) * 100
            print(f"  Processed {batch_idx + 1}/{total_batches} batches ({progress:.1f}%)")

    print("Concept similarity computation completed!")
    return torch.cat(all_text_sims), torch.cat(all_image_sims), all_decoded_texts  


def collate_fn(batch):
    images, texts, embeddings, categories = zip(*batch)  # Unzip batch into image, text, embedding, and category tuples
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    #processor = AutoProcessor.from_pretrained("zer0int/LongCLIP-GmP-ViT-L-14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Process images and text with CLIP processor
    #print(texts,images)
    inputs = processor(
        text=list(texts), 
        images=list(images), 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=77  # Fixes mismatch
    )
    # Add safe=0 label
    inputs["safe"] = torch.zeros(len(batch), dtype=torch.int64).to(device)  # Batch size unsafe labels

    # Add embeddings
    inputs["embedding"] = torch.stack(embeddings).to(device)

    # Add categories
    inputs["category"] = torch.tensor(categories, dtype=torch.int64).to(device)
    inputs["texts"] = list(texts)  # Convert tuple to list
    inputs["images"] = list(images)  


    # Debugging: Print processor output keys
    #print("Processor Output Keys:", inputs.keys())  # Should show: input_ids, attention_mask, pixel_values, safe, embedding, category
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs  # Move tensors to device


def main(unsafe_dataset,safe_dataset,concept_numbers):
    print("=" * 60)
    print("GENERATING DATALOADER WITH CONCEPT SIMILARITIES")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("CLIP model loaded successfully!")

#     processor = AutoProcessor.from_pretrained("zer0int/LongCLIP-GmP-ViT-L-14")
#     clip_model = AutoModelForZeroShotImageClassification.from_pretrained("zer0int/LongCLIP-GmP-ViT-L-14").to(device) 
    # Define concepts
    print(f"Loading {concept_numbers} concepts from concept.json...")
    label_path = "concept.json"  # Multiple labels
    with open(label_path, "r") as json_file:
        concept_dict = json.load(json_file)

    # Preprocess labels only once
    print("Processing concept labels...")
    merged_list = [item.strip("[]") for key, value in concept_dict.items() for item in value.split(", ")]
    sampled_strings = random.sample(merged_list, concept_numbers)
    merged_list = [random.choice(sampled_strings) for _ in range(concept_numbers)]

    print("Computing concept embeddings...")
    text_inputs1 = processor(text=merged_list, return_tensors="pt", padding=True).to(device)
    text_embeddings1 = clip_model.get_text_features(**text_inputs1)
    concept_embeddings = F.normalize(text_embeddings1, p=2, dim=1)  # Normalize once (1300,768)
    print(f"Concept embeddings computed: {concept_embeddings.shape}")

    # Define the number of samples to use from the unsafe dataset
    unsafe_sample_size = len(unsafe_dataset) // 2  # Use half (4000 if unsafe has 8000)
    safe_sample_size = len(safe_dataset)  # Keep all safe data
    
    print(f"Dataset sizes:")
    print(f"  Unsafe dataset: {len(unsafe_dataset)} samples")
    print(f"  Safe dataset: {len(safe_dataset)} samples")
    print(f"  Using {unsafe_sample_size} unsafe samples (50%)")

    # Randomly select a subset of the unsafe dataset
    print("Creating dataset subset...")
    unsafe_subset, _ = random_split(unsafe_dataset, [unsafe_sample_size, len(unsafe_dataset) - unsafe_sample_size])

    # Combine the datasets (using half of the unsafe dataset)
    combined_dataset = ConcatDataset([safe_dataset, unsafe_subset])
    print(f"Combined dataset created with {len(combined_dataset)} total samples")

    # Create a new dataloader
    print("Creating dataloader...")
    combined_dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Move concept embeddings to CPU temporarily if needed
    concept_embeddings = concept_embeddings.to("cpu")

    # Compute similarities for the whole dataset
    print("\nStarting concept similarity computation...")
    safe_text_sims, safe_image_sims, all_decoded_texts = compute_concept_similarities(combined_dataloader, clip_model, concept_embeddings,processor)

    # Move back to GPU if training on GPU
    print("Moving similarity tensors to GPU...")
    safe_text_sims = safe_text_sims.to(device)
    safe_image_sims = safe_image_sims.to(device)

    # Extract labels from the original dataloader
    print("Extracting category labels...")
    category_labels = torch.cat([batch["category"] for batch in combined_dataloader])  # (batch_size,)

    # Create dataset
    print("Creating final concept dataset...")
    concept_dataset = ConceptDataset(safe_text_sims, safe_image_sims, category_labels)
    
    print("=" * 60)
    print("DATALOADER GENERATION COMPLETED!")
    print(f"Final dataset size: {len(concept_dataset)} samples")
    print(f"Text similarities shape: {safe_text_sims.shape}")
    print(f"Image similarities shape: {safe_image_sims.shape}")
    print("=" * 60)
    
    return combined_dataset, concept_embeddings
