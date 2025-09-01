import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

class UnsafeVLMDataset_MMsafety(Dataset):
    def __init__(self, base_path, embedding_base_path, categories, processor, device="cuda"):
        self.device = device
        self.processor = processor
        self.base_path = base_path  # Base directory for unsafe dataset
        self.embedding_base_path = embedding_base_path  # Path for hidden state embeddings
        self.image_base_path = "/data/mmsafety/imgs"
        self.data = []  # Store (image_path, question_text, embedding, category) tuples

        # Create a category-to-index mapping (01 → 1, 02 → 2, etc.)
        category_map = {category: i+1 for i, category in enumerate(categories)}

        # Iterate over all categories and attack types
        for category in categories:
            category_label = category_map[category]  # Convert category name to numerical label

            for attack_type in ["attack_failure.json", "attack_success.json"]:
                json_path = os.path.join(base_path, "unsafe_input", "sample", category, attack_type)
                
                # Load JSON data
                if not os.path.exists(json_path):
                    print(f"Warning: {json_path} not found. Skipping {attack_type}.")
                    continue

                with open(json_path, "r") as f:
                    category_data = json.load(f)

                # Load embeddings for this category
                embedding_file = os.path.join(embedding_base_path, category, attack_type.replace(".json", ".pt"))
                embeddings = self.load_embeddings(embedding_file)

                # Create a mapping of embeddings by ID
                embedding_dict = {str(item["id"]): item["Rephrased_Question_hidden_states"].mean(dim=1).squeeze().numpy()
                                  for item in embeddings if "Rephrased_Question_hidden_states" in item}

                # Iterate over each entry in the JSON
                for item in category_data:
                    item_id = str(item["id"])  # Ensure ID is treated as a string for consistent matching

                    # Construct full image path
                    if "image_path" not in item:
                        print(f"Warning: Missing image_path in JSON entry. Skipping entry: {item}")
                        continue

                    image_path = os.path.join(self.image_base_path, item["image_path"].split("images/")[1])
                    question_text = item.get("Rephrased Question(SD)", "")
                    # Ensure the image exists before adding to dataset
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

# Custom `collate_fn` for dynamic padding and adding `safe=0`
def collate_fn(batch):
    images, texts, embeddings, categories = zip(*batch)  # Unzip batch into image, text, embedding, and category tuples

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

    # Debugging: Print processor output keys
    #print("Processor Output Keys:", inputs.keys())  # Should show: input_ids, attention_mask, pixel_values, safe, embedding, category

    return {k: v.to(device) for k, v in inputs.items()}  # Move tensors to device



def main(model,processor):
    # Load CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define correct paths
    base_path = "/"
    embedding_base_path = "/data/mmsafety/unsafe_input/weights"

    categories = [
        "01-Illegal_Activity",
        "02-HateSpeech",
        "03-Malware_Generation",
        "04-Physical_Harm",
        "05-EconomicHarm",
        "06-Fraud",
        "07-Sex",
        "08-Political_Lobbying",
        "09-Privacy_Violence",
        "10-Legal_Opinion",
        "11-Financial_Advice",
        "12-Health_Consultation",
        "13-Gov_Decision"
    ]

    # Create dataset and dataloader
    unsafe_dataset = UnsafeVLMDataset_MMsafety(base_path, embedding_base_path, categories, processor, device)
    unsafe_dataloader = DataLoader(unsafe_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    return unsafe_dataset,unsafe_dataloader
