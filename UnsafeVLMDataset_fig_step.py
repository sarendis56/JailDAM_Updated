import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

class UnsafeVLMDataset_fig_step(Dataset):
    def __init__(self, base_path, embedding_base_path, processor, device="cuda"):
        self.device = device
        self.processor = processor
        self.base_path = base_path  # Base directory for unsafe dataset
        self.embedding_base_path = embedding_base_path  # Path for hidden state embeddings
        self.data = []  # Store (image_path, question_text, embedding, category) tuples
        
        # Get all JSON files dynamically and create category mapping
        json_files = sorted([f for f in os.listdir(base_path) if f.endswith(".json")])
        self.category_map = {filename: i+1 for i, filename in enumerate(json_files)}  # Assign labels starting from 1
        
        print("Loaded Categories:", self.category_map)  # Debugging: Print category map
        #print((self.category_map))
        # Iterate over all JSON files
        for json_file in json_files:
            category_label = self.category_map[json_file]  # Get assigned label
            json_path = os.path.join(base_path, json_file)
            
            # Load JSON data
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found. Skipping.")
                continue

            with open(json_path, "r") as f:
                category_data = json.load(f)

            # Load corresponding embeddings
            embedding_file = os.path.join(embedding_base_path, json_file.replace(".json", ".pt"))
            embeddings = self.load_embeddings(embedding_file)

            # Create embedding dictionary based on ID
            embedding_dict = {str(item["id"]): item["Rephrased_Question_hidden_states"].mean(dim=1).squeeze().numpy()
                              for item in embeddings if "Rephrased_Question_hidden_states" in item}

            # Process each entry in the JSON file
            for item in category_data:
                item_id = str(item["id"])  # Ensure consistent string ID

                if "image_path" not in item:
                    print(f"Warning: Missing image_path in JSON entry. Skipping entry: {item}")
                    continue

                image_path = os.path.join(self.base_path, item["image_path"])
                question_text = item.get("Question", "")
                
                if os.path.exists(image_path):
                    embedding = embedding_dict.get(item_id, torch.zeros(4096))  # Default to zero if missing
                    self.data.append((image_path, question_text, embedding, category_label))
                else:
                    print(f"Warning: Image {image_path} not found, skipping.")

    def load_embeddings(self, file_path):
        """Load embeddings from a given .pt file and return them as a list."""
        if not os.path.exists(file_path):
            print(f"Warning: Embedding file {file_path} not found.")
            return []

        try:
            data = torch.load(file_path, map_location="cpu")
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, question_text, embedding, category = self.data[idx]

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB").resize((224, 224))

        return image, question_text, torch.tensor(embedding, dtype=torch.float32), torch.tensor(category, dtype=torch.int64)


# **Custom `collate_fn` for Unsafe Data**
def collate_fn(batch):
    images, texts, embeddings, categories = zip(*batch)  # Unzip batch

    # Process images and text with CLIP processor
    inputs = processor(
        text=list(texts), 
        images=list(images), 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=77  # Fixes mismatch
    )

    # Add `safe=0` label for unsafe data
    inputs["safe"] = torch.zeros(len(batch), dtype=torch.int64).to(device)  

    # Add embeddings and categories
    inputs["embedding"] = torch.stack(embeddings).to(device)
    inputs["category"] = torch.tensor(categories, dtype=torch.int64).to(device)

    return {k: v.to(device) for k, v in inputs.items()}  # Move tensors to device

def main(model,processor):
    # **Load CLIP model and processor**
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # **Define paths**
    base_path = "/data/fig_step"
    embedding_base_path = "/data/mmsafety/unsafe_input/weights"

    # **Create Unsafe Dataset and DataLoader**
    unsafe_dataset = UnsafeVLMDataset_fig_step(base_path, embedding_base_path, processor, device)
    unsafe_dataloader = DataLoader(unsafe_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    return unsafe_dataset,unsafe_dataloader

