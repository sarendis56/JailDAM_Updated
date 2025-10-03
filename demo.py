#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoProcessor,
    AutoModelForZeroShotImageClassification,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# External modules provided by your environment
from UnsafeVLMDataset_28k import main as main_unsafe_28k
from UnsafeVLMDataset_MMsafety import main as main_unsafe_mmsafety
from UnsafeVLMDataset_fig_step import main as main_unsafe_fig_step
from VLMDataset_mmvet import main as main_safe_mmvet
from VLMDataset_medical_vqa import main as main_safe_medical_vqa
from VLMDataset_alpaca import main as main_safe_alpaca
from generate_dataloader import main as generate_dataloader
from memory_network import MemoryNetwork


# -------------------------
# Dataset loader
# -------------------------
def load_selected_datasets(unsafe_name, safe_name, model, processor, test_name=None, max_test_samples=1000):
    unsafe_datasets = {
        "UnsafeVLMDataset_28k": main_unsafe_28k,
        "UnsafeVLMDataset_MMsafety": main_unsafe_mmsafety,
        "UnsafeVLMDataset_fig_step": main_unsafe_fig_step,
    }
    safe_datasets = {
        "VLMDataset_mmvet": main_safe_mmvet,
        "VLMDataset_medical_vqa": main_safe_medical_vqa
    }
    if unsafe_name not in unsafe_datasets or safe_name not in safe_datasets:
        raise ValueError(
            f"Invalid dataset names. Choose from: {list(unsafe_datasets.keys())} (unsafe) and {list(safe_datasets.keys())} (safe)"
        )
    
    unsafe_dataset, unsafe_dataloader = unsafe_datasets[unsafe_name](model, processor)
    safe_dataset, safe_dataloader = safe_datasets[safe_name](model, processor)
    
    # Load test dataset if specified
    test_dataset, test_dataloader = None, None
    if test_name and test_name in safe_datasets:
        print(f"Loading test dataset: {test_name}")
        test_dataset, test_dataloader = safe_datasets[test_name](model, processor, max_samples=max_test_samples)
        print(f"Loaded {test_name} with {len(test_dataloader.dataset)} samples for testing.")
    
    return unsafe_dataset, unsafe_dataloader, safe_dataset, safe_dataloader, test_dataset, test_dataloader


# -------------------------
# Collate function
# -------------------------
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
    inputs["safe"] = torch.zeros(len(batch), dtype=torch.int64, device=device)
    inputs["embedding"] = torch.stack(embeddings).to(device)
    inputs["category"] = torch.tensor(categories, dtype=torch.int64, device=device)
    inputs["texts"] = list(texts)
    inputs["images"] = list(images)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


# -------------------------
# Autoencoder
# -------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):  # noqa: D401
        return self.decoder(self.encoder(x))


# -------------------------
# Concept embedding updater
# -------------------------
def update_concept_embeddings(
    concept_embeddings,
    concept_frequency_total,
    attention_features_text,
    attention_features_image,
    threshold,
    top_k=5,
    device="cuda",
):
    def update_embeddings(features, concept_slice):
        softmax_probs = torch.nn.functional.softmax(features, dim=-1)
        max_probs, indices = torch.topk(softmax_probs, top_k, dim=-1)
        updates = []

        for i in range(features.shape[0]):
            if max_probs[i, 0] > threshold:
                freq_tensor = torch.tensor(list(concept_frequency_total.values()), device=device)
                min_freq_idx = torch.argmin(freq_tensor).item()

                top_k_concepts = concept_embeddings[indices[i], concept_slice.start:concept_slice.stop]
                weighted_sum = (max_probs[i].unsqueeze(-1) * top_k_concepts).sum(dim=0)
                new_concept = features[i] - weighted_sum

                updates.append((min_freq_idx, concept_slice.start, concept_slice.stop, new_concept.detach().clone()))
                concept_frequency_total[min_freq_idx] = torch.max(freq_tensor).item() + 1

        for min_freq_idx, start, stop, new_concept in updates:
            concept_embeddings[min_freq_idx, start:stop] = new_concept

    update_embeddings(attention_features_text, slice(0, 768))
    update_embeddings(attention_features_image, slice(768, 1536))
    return concept_embeddings, concept_frequency_total


# -------------------------
# Visualization Functions
# -------------------------
def create_score_distribution_plot(all_scores, all_labels, dataset_names, save_path="score_distribution.pdf"):
    """Create a PDF visualization of score distributions for each dataset"""
    # Ensure numpy arrays for correct boolean masking behavior
    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)
    dataset_names = np.asarray(dataset_names)

    plt.style.use('default')
    # Double font sizes
    base_fs = plt.rcParams.get('font.size', 12)
    fs = plt.rcParams.get('font.size', 18)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get unique datasets
    unique_datasets = np.unique(dataset_names)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Histogram comparison
    for i, dataset in enumerate(unique_datasets):
        mask = dataset_names == dataset
        scores = all_scores[mask]
        labels = all_labels[mask]
        
        # Determine if this dataset should be benign (0) or malicious (1)
        expected_label = int(labels[0]) if labels.size > 0 else 0
        dataset_type = "Benign" if expected_label == 0 else "Malicious"
        # Legend wants: DatasetName(Property). If the name already carries a property in parentheses,
        # keep it as-is to avoid duplication.
        legend_label = dataset if "(" in str(dataset) else f"{dataset} ({dataset_type})"
        
        ax1.hist(scores, bins=30, alpha=0.7, label=legend_label, 
                color=colors[i % len(colors)], density=True)
    
    ax1.set_xlabel('Reconstruction Error Score', fontsize=fs)
    ax1.set_ylabel('Density', fontsize=fs)
    ax1.set_title('Score Distribution Comparison Across Datasets', fontsize=fs)
    ax1.legend(fontsize=base_fs)
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    box_data = []
    box_labels = []
    box_colors = []
    
    for i, dataset in enumerate(unique_datasets):
        mask = dataset_names == dataset
        scores = all_scores[mask]
        labels = all_labels[mask]
        
        expected_label = int(labels[0]) if labels.size > 0 else 0
        dataset_type = "Benign" if expected_label == 0 else "Malicious"
        
        box_data.append(scores)
        # For x-axis of the second plot, show only dataset name (no property) and keep horizontal
        ds_name_only = str(dataset)
        if "(" in ds_name_only and ")" in ds_name_only:
            # strip trailing parenthetical
            ds_name_only = ds_name_only.split("(")[0].strip()
        box_labels.append(ds_name_only)
        box_colors.append(colors[i % len(colors)])
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Reconstruction Error Score', fontsize=fs)
    ax2.set_title('Score Distribution Box Plot', fontsize=fs)
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=0, fontsize=fs)
    ax2.tick_params(axis='both', labelsize=fs)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Score distribution plot saved to: {save_path}")

def print_experiment_configuration(experiment_name, train_samples, val_samples, test_samples_info, description):
    """Print detailed experiment configuration"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    print(f"Description: {description}")
    print(f"\nDataset Configuration:")
    print(f"  Training: {train_samples} samples")
    print(f"  Validation: {val_samples} samples")
    print(f"  Testing:")
    for test_name, test_count, test_type in test_samples_info:
        print(f"    - {test_name}: {test_count} samples ({test_type})")
    print(f"{'='*80}")

# -------------------------
# Evaluation
# -------------------------
def evaluate_autoencoder_combined(
    dataloaders,
    concept_embeddings,
    autoencoder,
    device,
    concept_frequency_total,
    adapt_embeddings=True,
):
    autoencoder.eval()
    # Clone locally to avoid unintended in-place mutation unless caller uses the returned value
    concept_embeddings = concept_embeddings.clone().to(device)

    all_labels, all_scores = [], []
    modified_batches, total_inputs, total_time = [], 0, 0.0

    with torch.no_grad():
        for dataloader in dataloaders:
            start_time = time.time()
            for batch in dataloader:
                t0 = time.time()

                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["category"].cpu().numpy()
                binary_labels = np.where(labels > 0, 1, 0)

                _, _, text_embedding, image_embedding, _, _ = memory_network.forward(
                    text_input_ids=input_ids, text_attention_mask=attention_mask, image_pixel_values=pixel_values
                )

                sim_img = image_embedding @ concept_embeddings[:, :768].T
                sim_txt = text_embedding @ concept_embeddings[:, 768:].T
                attention_features = torch.cat((sim_txt, sim_img), dim=-1)

                if adapt_embeddings:
                    concept_embeddings, concept_frequency_total = update_concept_embeddings(
                        concept_embeddings, concept_frequency_total, text_embedding, image_embedding, 0.0004, device=device
                    )

                recon = autoencoder(attention_features)
                recon_err = torch.mean((recon - attention_features) ** 2, dim=-1)

                all_labels.extend(binary_labels)
                all_scores.extend(recon_err.cpu().numpy())

                batch["reconstruction_error"] = recon_err.cpu().numpy()
                modified_batches.append(batch)

                total_time += (time.time() - t0)
                total_inputs += len(labels)

            print(f"Execution Time for Dataloader: {time.time() - start_time:.4f} seconds")

    avg_time = total_time / total_inputs if total_inputs > 0 else 0.0
    print(f"Average Processing Time per Input: {avg_time:.6f} seconds")

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    auroc = roc_auc_score(all_labels, all_scores)
    aupr = average_precision_score(all_labels, all_scores)

    best_f1, best_threshold = 0.0, 0.0
    for th in np.linspace(all_scores.min(), all_scores.max(), 100):
        preds = (all_scores >= th).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=1)
        if f1 > best_f1:
            best_f1, best_threshold = f1, th

    final_preds = (all_scores >= best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, final_preds, average="binary", zero_division=1)

    for batch in modified_batches:
        batch["prediction"] = (batch["reconstruction_error"] >= best_threshold).astype(int)

    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "Best Threshold": best_threshold,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Average Processing Time per Input": avg_time,
        "Modified Batches": modified_batches,
        "Concept Embeddings": concept_embeddings.detach().clone(),
    }


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model for dataset preprocessing
    ds_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    ds_model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")

    # Load datasets
    unsafe_dataset_name = "UnsafeVLMDataset_MMsafety"
    safe_dataset_name = "VLMDataset_mmvet"
    test_benign_name = "VLMDataset_medical_vqa"  # VQA-RAD for unseen benign testing
    max_test_samples = 500  # Cap VQA-RAD to 500

    unsafe_dataset, unsafe_dataloader, safe_dataset, safe_dataloader, test_benign_dataset, test_benign_dataloader = load_selected_datasets(
        unsafe_dataset_name, safe_dataset_name, ds_model, ds_processor, test_benign_name, max_test_samples
    )
    # Load additional text-only benign instructions dataset (instruction tuning style)
    text_only_max = 500
    text_only_dataset, text_only_dataloader = main_safe_alpaca(ds_model, ds_processor, max_samples=text_only_max)
    
    print(f"Loaded {unsafe_dataset_name} with {len(unsafe_dataloader.dataset)} samples.")
    print(f"Loaded {safe_dataset_name} with {len(safe_dataloader.dataset)} samples.")
    if test_benign_dataset and test_benign_dataloader:
        print(f"Loaded {test_benign_name} with {len(test_benign_dataloader.dataset)} samples for unseen benign testing.")
    print(f"Loaded Text-Only Instructions with {len(text_only_dataloader.dataset)} samples for unseen benign testing.")

    samples_per_category = 100
    num_categories = 13
    combined_dataset, concept_embeddings_ori = generate_dataloader(
        unsafe_dataset, safe_dataset, samples_per_category * num_categories
    )

    print("Combined dataset created")

    # CLIP for memory network
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    _ = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")  # kept for parity with original environment

    print("CLIP Model loaded")

    criterion = nn.MSELoss()
    batch_size = 8

    for learning_rate_concept in [0.5]:
        total_per_category = 100
        categories_to_remove = [1, 2, 3, 4, 5]
        num_categories_train = 8

        rows_to_remove = []
        for cat_idx in categories_to_remove:
            start = cat_idx * total_per_category
            rows_to_remove.extend(range(start, start + total_per_category))

        keep_indices = sorted(set(range(concept_embeddings_ori.shape[0])) - set(rows_to_remove))
        reduced_concept_embeddings = concept_embeddings_ori[keep_indices]
        print(reduced_concept_embeddings.shape)

        memory_network = MemoryNetwork(
            clip_model=clip_model,
            concept_embeddings=reduced_concept_embeddings,
            device=device,
        ).to(device)

        concept_embeddings = reduced_concept_embeddings.clone().detach().requires_grad_(True)
        _concept_optimizer = optim.Adam([concept_embeddings], lr=learning_rate_concept)  # created (same as original), later replaced

        # Split data
        safe_only = [d for d in combined_dataset if d[-1] == 0]
        train_size = int(0.8 * len(safe_only))
        val_size = len(safe_only) - train_size
        train_dataset, val_dataset = random_split(safe_only, [train_size, val_size])
        unsafe_only = [d for d in combined_dataset if d[-1] != 0]

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        ood_dataloader = DataLoader(unsafe_only, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Unsafe test samples: {len(unsafe_only)}")
        if test_benign_dataloader:
            print(f"Unseen benign test samples: {len(test_benign_dataloader.dataset)}")

        num_epochs, learning_rate, _, update_threshold, top_k_update = 5, 0.001, 8, 10, 50

        # Autoencoder input: num_categories_train * samples_per_category * 2
        autoencoder = Autoencoder(num_categories_train * samples_per_category * 2).to(device)
        autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

        # 1536-dim learnable concepts (text+image)
        concept_embeddings = torch.nn.Parameter(torch.cat((concept_embeddings, concept_embeddings), dim=-1).to(device))
        concept_optimizer = optim.Adam([concept_embeddings], lr=learning_rate)
        similarity_loss_fn = nn.CosineEmbeddingLoss()
        concept_frequency_total = {i: 0 for i in range(len(concept_embeddings))}

        print("=" * 60)
        print("Starting JailDAM Training")
        print("=" * 60)
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_benign_dataloader.dataset) if test_benign_dataloader else len(unsafe_only)}")
        print(f"Test dataset: {test_benign_name if test_benign_dataloader else 'Unsafe Data (OOD)'}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            autoencoder.train()
            total_loss = 0.0
            concept_loss_total = 0.0

            concept_frequency = {i: 0 for i in range(len(concept_embeddings))}
            concept_match_accumulator = {i: [] for i in range(len(concept_embeddings))}

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            batch_count = 0
            for batch in train_dataloader:
                autoencoder_optimizer.zero_grad()
                concept_optimizer.zero_grad()

                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                _, _, text_emb, img_emb, _, _ = memory_network.forward(
                    text_input_ids=input_ids, text_attention_mask=attention_mask, image_pixel_values=pixel_values
                )

                sim_img = img_emb @ concept_embeddings[:, :768].T
                sim_txt = text_emb @ concept_embeddings[:, 768:].T

                features = torch.cat((sim_txt, sim_img), dim=-1)
                recon = autoencoder(features)
                ae_loss = criterion(recon, features)

                top5_img = sim_img.topk(5, dim=-1).indices
                top5_txt = sim_txt.topk(5, dim=-1).indices
                for i in range(features.size(0)):
                    matched = torch.cat((text_emb[i], img_emb[i]), dim=-1)
                    for idx in top5_img[i].tolist() + top5_txt[i].tolist():
                        concept_frequency[idx] += 1
                        concept_frequency_total[idx] += 1
                        concept_match_accumulator[idx].append(matched.detach())

                ae_loss.backward(retain_graph=True)
                autoencoder_optimizer.step()
                total_loss += ae_loss.item()
                
                batch_count += 1
                if batch_count % 10 == 0:  # Print progress every 10 batches
                    print(f"  Batch {batch_count}/{len(train_dataloader)} - Loss: {ae_loss.item():.4f}")

            sorted_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)
            top_k_concepts = [idx for idx, freq in sorted_concepts[:top_k_update] if freq > update_threshold]

            if top_k_concepts:
                concept_optimizer.zero_grad()
                concept_loss = torch.tensor(0.0, device=device, requires_grad=True)

                for idx in top_k_concepts:
                    if concept_match_accumulator[idx]:
                        matches = torch.stack(concept_match_accumulator[idx])
                        mean_vec = matches.mean(dim=0).detach()
                        var_vec = ((matches - mean_vec) ** 2).mean(dim=0).detach() if matches.size(0) > 1 else torch.zeros_like(mean_vec)

                        concept_vec = concept_embeddings[idx].unsqueeze(0)
                        target = torch.ones(1, device=device)
                        loss_val = similarity_loss_fn(concept_vec, mean_vec.unsqueeze(0), target)
                        concept_loss = concept_loss + 5.0 * loss_val * (1 + var_vec.mean())

                (concept_loss / len(top_k_concepts)).backward()
                concept_optimizer.step()
                concept_loss_total += concept_loss.item()

            avg_loss = total_loss / len(train_dataloader)
            avg_concept_loss = concept_loss_total / len(train_dataloader)
            print(f"  Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"    Average Autoencoder Loss: {avg_loss:.4f}")
            print(f"    Average Concept Loss: {avg_concept_loss:.4f}")
            print(f"    Updated Concepts: {len(top_k_concepts)}")
            print(f"    Total Batches Processed: {batch_count}")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("Only top-K most frequently used unsafe concept embeddings were updated.")
        print("=" * 60)

    # Snapshot baseline states for evaluation experiments
    baseline_concept_embeddings = concept_embeddings.detach().clone()
    baseline_concept_frequency_total = {k: v for k, v in concept_frequency_total.items()}

    # Evaluate on multiple test sets
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Test 1: Validation + Unsafe (original approach)
    print_experiment_configuration(
        experiment_name="Original Jailbreak Detection",
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        test_samples_info=[
            ("Validation (MM-Vet)", len(val_dataset), "Benign"),
            ("Unsafe (MMSafety)", len(unsafe_only), "Malicious")
        ],
        description="Baseline experiment using the original JailDAM approach: train on safe data, test on safe validation + unsafe data"
    )
    
    # Fresh copies for experiment 1 (allowing test-time adaptation within this experiment only)
    concept_embeddings_exp1 = baseline_concept_embeddings.clone().to(device)
    concept_frequency_total_exp1 = {k: v for k, v in baseline_concept_frequency_total.items()}
    combined_dataloaders_original = [val_dataloader, ood_dataloader]
    results_original = evaluate_autoencoder_combined(
        combined_dataloaders_original,
        concept_embeddings_exp1,
        autoencoder,
        device,
        concept_frequency_total_exp1,
        adapt_embeddings=False,
    )
    print(f"\nResults:")
    print(f"   AUROC: {results_original['AUROC']:.4f}, AUPR: {results_original['AUPR']:.4f}")
    print(f"   F1 Score: {results_original['F1 Score']:.4f}, Precision: {results_original['Precision']:.4f}, Recall: {results_original['Recall']:.4f}")
    
    # Generate score distribution plot for the first (original) experiment
    datasets_info_original = [
        ("Validation (MM-Vet)", val_dataloader, 0),
        ("Unsafe (MMSafety)", ood_dataloader, 1)
    ]
    all_scores_orig = []
    all_labels_orig = []
    dataset_names_orig = []
    
    autoencoder.eval()
    with torch.no_grad():
        for dataset_name, dataloader, expected_label in datasets_info_original:
            if len(dataloader.dataset) == 0:
                continue
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                _, _, text_embedding, image_embedding, _, _ = memory_network.forward(
                    text_input_ids=input_ids, text_attention_mask=attention_mask, image_pixel_values=pixel_values
                )
                # Use the exact embeddings from the evaluation (no adaptation)
                ce_eval = results_original["Concept Embeddings"]
                sim_img = image_embedding @ ce_eval[:, :768].T
                sim_txt = text_embedding @ ce_eval[:, 768:].T
                attention_features = torch.cat((sim_txt, sim_img), dim=-1)
                recon = autoencoder(attention_features)
                recon_err = torch.mean((recon - attention_features) ** 2, dim=-1)
                
                scores_np = recon_err.cpu().numpy()
                all_scores_orig.extend(scores_np)
                all_labels_orig.extend([expected_label] * len(scores_np))
                dataset_names_orig.extend([dataset_name] * len(scores_np))

    # -------------------------
    # Diagnostics for figure vs metrics
    # -------------------------
    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        y_true_plot = np.array(all_labels_orig)
        y_score_plot = np.array(all_scores_orig)
        if y_true_plot.size > 0:
            auroc_plot = roc_auc_score(y_true_plot, y_score_plot)
            print(f"[Diagnostics] AUROC from plot arrays: {auroc_plot:.6f}")
            # Percentile overlap check
            benign_scores = y_score_plot[y_true_plot == 0]
            malicious_scores = y_score_plot[y_true_plot == 1]
            if benign_scores.size > 0 and malicious_scores.size > 0:
                q_b = np.quantile(benign_scores, [0.5/100, 0.5, 99.5/100])
                q_m = np.quantile(malicious_scores, [0.5/100, 0.5, 99.5/100])
                print(
                    f"[Diagnostics] Benign percentiles (0.5%, 50%, 99.5%): {q_b[0]:.4f}, {q_b[1]:.4f}, {q_b[2]:.4f}"
                )
                print(
                    f"[Diagnostics] Malicious percentiles (0.5%, 50%, 99.5%): {q_m[0]:.4f}, {q_m[1]:.4f}, {q_m[2]:.4f}"
                )
                print(
                    f"[Diagnostics] Benign max vs Malicious min: {benign_scores.max():.4f} vs {malicious_scores.min():.4f}"
                )
    except Exception as e:
        print(f"[Diagnostics] Error computing diagnostics: {e}")

    timestamp_orig = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path_orig = f"jaildam_score_distribution_original_{timestamp_orig}.pdf"
    create_score_distribution_plot(np.array(all_scores_orig), np.array(all_labels_orig), np.array(dataset_names_orig), pdf_path_orig)
    
    # Test 2: All test sets combined with detailed analysis
    if test_benign_dataloader:
        print_experiment_configuration(
            experiment_name="Robust Jailbreak Detection with Distribution Shift",
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
            test_samples_info=[
                ("Validation (MM-Vet)", len(val_dataset), "Benign (Seen)"),
                ("Unsafe (MMSafety)", len(unsafe_only), "Malicious (Seen)"),
                ("VQA-RAD (Medical)", len(test_benign_dataloader.dataset), "Benign (Unseen)"),
                ("Text-Only Instructions", len(text_only_dataloader.dataset), "Benign (Unseen)")
            ],
            description="Robustness test: train on safe data, test on safe validation + unsafe data + unseen benign data from different domain (medical) to test distribution shift robustness"
        )
        
        # Fresh copies for experiment 2 (allowing test-time adaptation within this experiment only)
        concept_embeddings_exp2 = baseline_concept_embeddings.clone().to(device)
        concept_frequency_total_exp2 = {k: v for k, v in baseline_concept_frequency_total.items()}
        combined_dataloaders_all = [val_dataloader, ood_dataloader, test_benign_dataloader, text_only_dataloader]
        results_all = evaluate_autoencoder_combined(
            combined_dataloaders_all,
            concept_embeddings_exp2,
            autoencoder,
            device,
            concept_frequency_total_exp2,
            adapt_embeddings=False,
        )
        print(f"\nResults:")
        print(f"   AUROC: {results_all['AUROC']:.4f}, AUPR: {results_all['AUPR']:.4f}")
        print(f"   F1 Score: {results_all['F1 Score']:.4f}, Precision: {results_all['Precision']:.4f}, Recall: {results_all['Recall']:.4f}")
        
        # Detailed per-dataset analysis
        print("\n" + "="*80)
        print("DETAILED PER-DATASET ANALYSIS")
        print("="*80)
        
        # Analyze each dataset separately
        datasets_info = [
            ("Validation (Safe)", val_dataloader, 0),
            ("Unsafe (Malicious)", ood_dataloader, 1), 
            ("VQA-RAD (Unseen Benign)", test_benign_dataloader, 0),
            ("Text-Only Instructions (Unseen Benign)", text_only_dataloader, 0)
        ]
        
        all_scores = []
        all_labels = []
        dataset_names = []
        
        for dataset_name, dataloader, expected_label in datasets_info:
            print(f"\n{dataset_name}:")
            scores = []
            labels = []
            
            autoencoder.eval()
            with torch.no_grad():
                for batch in dataloader:
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    
                    _, _, text_embedding, image_embedding, _, _ = memory_network.forward(
                        text_input_ids=input_ids, text_attention_mask=attention_mask, image_pixel_values=pixel_values
                    )
                    
                    # Use the exact embeddings from the evaluation (no adaptation)
                    ce_eval2 = results_all["Concept Embeddings"]
                    sim_img = image_embedding @ ce_eval2[:, :768].T
                    sim_txt = text_embedding @ ce_eval2[:, 768:].T
                    attention_features = torch.cat((sim_txt, sim_img), dim=-1)
                    
                    recon = autoencoder(attention_features)
                    recon_err = torch.mean((recon - attention_features) ** 2, dim=-1)
                    
                    scores.extend(recon_err.cpu().numpy())
                    labels.extend([expected_label] * len(recon_err))
            
            # Handle empty datasets gracefully
            if len(scores) == 0:
                print("  Samples: 0")
                print("  Mean Score: n/a")
                print("  Std Score: n/a")
                print("  Min Score: n/a")
                print("  Max Score: n/a")
                print("  Median Score: n/a")
                continue

            scores = np.array(scores)
            labels = np.array(labels)
            
            all_scores.extend(scores)
            all_labels.extend(labels)
            dataset_names.extend([dataset_name] * len(scores))
            
            print(f"  Samples: {len(scores)}")
            print(f"  Mean Score: {scores.mean():.4f}")
            print(f"  Std Score: {scores.std():.4f}")
            print(f"  Min Score: {scores.min():.4f}")
            print(f"  Max Score: {scores.max():.4f}")
            print(f"  Median Score: {np.median(scores):.4f}")
            
            # Score distribution bins
            print(f"  Score Distribution:")
            bins = np.linspace(scores.min(), scores.max(), 11)
            hist, _ = np.histogram(scores, bins=bins)
            for i in range(len(hist)):
                bin_start = bins[i]
                bin_end = bins[i+1]
                count = hist[i]
                bar = "█" * min(count, 50)  # Limit bar length
                print(f"    [{bin_start:.2f}, {bin_end:.2f}): {count:3d} {bar}")
        
        # Create PDF visualization if we have any scores
        if len(all_scores) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = f"jaildam_score_distribution_{timestamp}.pdf"
            create_score_distribution_plot(all_scores, all_labels, dataset_names, pdf_path)
        
        # Overall confusion analysis
        print(f"\n" + "="*80)
        print("CONFUSION ANALYSIS")
        print("="*80)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        dataset_names = np.array(dataset_names)
        
        # Find optimal threshold
        best_f1, best_threshold = 0.0, 0.0
        for th in np.linspace(all_scores.min(), all_scores.max(), 100):
            preds = (all_scores >= th).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=1)
            if f1 > best_f1:
                best_f1, best_threshold = f1, th
        
        print(f"Optimal Threshold: {best_threshold:.4f}")
        print(f"Best F1 Score: {best_f1:.4f}")
        
        # Per-dataset performance at optimal threshold
        for dataset_name, dataloader, expected_label in datasets_info:
            dataset_scores = all_scores[dataset_names == dataset_name]
            dataset_labels = all_labels[dataset_names == dataset_name]
            if len(dataset_scores) == 0:
                print(f"\n{dataset_name}: no samples, skipping confusion analysis.")
                continue
            dataset_preds = (dataset_scores >= best_threshold).astype(int)
            
            correct = (dataset_preds == dataset_labels).sum()
            total = len(dataset_preds)
            accuracy = correct / total
            
            print(f"\n{dataset_name} at threshold {best_threshold:.4f}:")
            print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
            print(f"  Predicted as Malicious: {(dataset_preds == 1).sum()}/{total}")
            print(f"  Predicted as Benign: {(dataset_preds == 0).sum()}/{total}")
            
            if expected_label == 0:  # Should be benign
                false_positive_rate = (dataset_preds == 1).sum() / total
                print(f"  False Positive Rate: {false_positive_rate:.4f}")
            else:  # Should be malicious
                true_positive_rate = (dataset_preds == 1).sum() / total
                print(f"  True Positive Rate: {true_positive_rate:.4f}")
    else:
        print("\n2. VQA-RAD dataset not available, skipping comprehensive analysis.")
    
    print(f"\nAverage Processing Time per Input: {results_original['Average Processing Time per Input']:.6f} seconds")
    
    # Final Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print("This evaluation tested the JailDAM framework's robustness to distribution shift:")
    print("1. Original approach: Train on safe data, test on safe + unsafe data")
    print("2. Robustness test: Train on safe data, test on safe + unsafe + unseen benign data")
    print("\nKey insights:")
    print("- The method's ability to distinguish between benign and malicious samples")
    print("- How distribution shift affects detection performance")
    print("- Whether unseen benign data is misclassified as malicious")
    print("- Score distributions reveal decision-making patterns")
    print("="*80)


# In[ ]:
