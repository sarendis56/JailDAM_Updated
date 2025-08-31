from UnsafeVLMDataset_28k import main as main_unsafe_28k
from UnsafeVLMDataset_MMsafety import main as main_unsafe_mmsafety
from UnsafeVLMDataset_fig_step import main as main_unsafe_fig_step
from VLMDataset_vlguard import main as main_safe_vlguard
from VLMDataset_mmvet import main as main_safe_mmvet
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, CLIPModel, CLIPProcessor
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import numpy as np
import time

# Collate Function
def collate_fn(batch):
    images, texts, embeddings, categories = zip(*batch)
    processor =  CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    inputs = processor(
        text=list(texts), 
        images=list(images), 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=77
    )
    inputs["safe"] = torch.zeros(len(batch), dtype=torch.int64)
    inputs["embedding"] = torch.stack(embeddings)
    inputs["category"] = torch.tensor(categories, dtype=torch.int64)
    inputs["texts"] = list(texts)
    inputs["images"] = list(images)
    return inputs


# Load datasets
def load_selected_datasets(unsafe_name, safe_name, model, processor):
    unsafe_datasets = {
        "UnsafeVLMDataset_28k": main_unsafe_28k,
        "UnsafeVLMDataset_MMsafety": main_unsafe_mmsafety,
        "UnsafeVLMDataset_fig_step": main_unsafe_fig_step
    }
    safe_datasets = {
        "VLMDataset_mmvet": main_safe_mmvet,
        "VLMDataset_vlguard": main_safe_vlguard
    }
    unsafe_dataset, unsafe_dataloader = unsafe_datasets[unsafe_name](model, processor)
    safe_dataset, safe_dataloader = safe_datasets[safe_name](model, processor)
    return unsafe_dataset, unsafe_dataloader, safe_dataset, safe_dataloader


# Concept Embedding Updater
def update_concept_embeddings(concept_embeddings, concept_frequency_total, attention_features_text, attention_features_image, text_embedding,image_embedding, threshold, DEVICE,top_k=5):
    def update(features, embedding,slice_range):
        softmax_probs = torch.nn.functional.softmax(features, dim=-1)
        max_probs, indices = torch.topk(softmax_probs, top_k, dim=-1)
        for i in range(features.shape[0]):
            if max_probs[i, 0] > threshold:

                freq_tensor = torch.tensor(list(concept_frequency_total.values()), device=DEVICE)
                min_idx = torch.argmin(freq_tensor).item()
                top_k_concepts = concept_embeddings[indices[i], :]
                weighted_sum = (max_probs[i].unsqueeze(-1) * top_k_concepts).sum(dim=0)
                new_concept = embedding[i] - weighted_sum
                concept_embeddings[min_idx,:] = new_concept.detach().clone()
                assert isinstance(min_idx, int), f"min_idx is not int: {min_idx}"
                assert 0 <= min_idx < concept_embeddings.shape[0], (
                    f"min_idx {min_idx} out of bounds for concept_embeddings shape {concept_embeddings.shape}"
                )
                assert slice_range.stop <= concept_embeddings.shape[1], (
                    f"Slice {slice_range} exceeds embedding dim {concept_embeddings.shape[1]}"
                )
                concept_frequency_total[min_idx] = torch.max(freq_tensor).item() + 1
                #concept_frequency_total[min_idx] = max(list(concept_frequency_total.values()))
                
    update(attention_features_text, text_embedding,slice(0, 768))
    update(attention_features_image,image_embedding, slice(0, 768))
    return concept_embeddings, concept_frequency_total


# Evaluation Function
def evaluate_autoencoder_combined(dataloaders, concept_embeddings, autoencoder, concept_frequency_total, DEVICE,memory_network, threshold=0.9):
    autoencoder.eval()  # Set model to evaluation mode
    concept_embeddings = concept_embeddings.to(DEVICE)  # Move concept embeddings to GPU

    all_labels, all_scores = [], []
    modified_batches = []  # Store modified batches
    total_inputs = 0  # Track number of inputs
    total_time = 0  # Track total execution time

    with torch.no_grad():
        for dataloader in dataloaders:
            start_time = time.time()
            for batch in dataloader:
                batch_start_time = time.time()  # Start time for batch
                
                pixel_values = batch["pixel_values"].to(DEVICE)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["category"].cpu().numpy()
                
                binary_labels = np.where(labels > 0, 1, 0)
            
                # Forward pass through memory network
                logits, memory_output, text_embedding, image_embedding, _, _ = memory_network.forward(
                    text_input_ids=input_ids, text_attention_mask=attention_mask, image_pixel_values=pixel_values
                )

                # Compute similarity scores
                similarity_image = torch.matmul(image_embedding, concept_embeddings.T)
                similarity_text = torch.matmul(text_embedding, concept_embeddings.T)
                
                # Concatenate text and image similarity scores
                attention_features = torch.cat((similarity_text, similarity_image), dim=-1)

                concept_embeddings, concept_frequency_total = update_concept_embeddings(
                    concept_embeddings, concept_frequency_total, similarity_text, similarity_image,text_embedding,image_embedding,0.0004,DEVICE
                )
                
                # Autoencoder reconstruction
                reconstructed_attention = autoencoder(attention_features)
                reconstruction_error = torch.mean((reconstructed_attention - attention_features) ** 2, dim=-1)

                all_labels.extend(binary_labels)
                all_scores.extend(reconstruction_error.cpu().numpy())

                # Attach predictions to the batch
                batch["reconstruction_error"] = reconstruction_error.cpu().numpy()
                modified_batches.append(batch)

                batch_end_time = time.time()  # End time for batch
                batch_time = batch_end_time - batch_start_time
                total_time += batch_time
                total_inputs += len(labels)  # Count number of inputs in batch

            end_time = time.time()  # End timing
            execution_time = end_time - start_time
            print(f"Execution Time for Dataloader: {execution_time:.4f} seconds")  # Display execution time

    # Compute average processing time per input
    avg_time_per_input = total_time / total_inputs if total_inputs > 0 else 0
    print(f"Average Processing Time per Input: {avg_time_per_input:.6f} seconds")

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Compute AUROC & AUPR
    auroc = roc_auc_score(all_labels, all_scores)
    aupr = average_precision_score(all_labels, all_scores)

    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "scores":all_scores,
        "labels":all_labels
    }

