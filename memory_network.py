import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class MemoryNetwork(nn.Module):
    def __init__(self, clip_model, concept_embeddings, device, embedding_dim=768, max_memory_size=1300, num_classes=14, learning_rate=0.001):
        """
        Memory-based classification model with soft attention and memory update.
        - Uses soft attention over stored memory concepts.
        - Learns memory through gradient-based updates.
        - Classifies input text/image using an MLP.
        """
        super(MemoryNetwork, self).__init__()

        self.clip_model = clip_model
        self.device = device
        self.embedding_dim = embedding_dim
        self.max_memory_size = max_memory_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # **Initialize Trainable Memory**
        #self.memory_concepts = nn.Parameter(torch.cat((concept_embeddings, concept_embeddings), dim=-1)) 
        self.memory_concepts = nn.Parameter(concept_embeddings) # Shape: (1300, 1536)

        # **MLP Classifier (2-layer)**
        self.mlp = nn.Sequential(
            nn.Linear(1536, 512),  # Concatenated memory output
            nn.ReLU(),
            nn.Linear(512, self.num_classes)  # 14-class classification
        )
        
    def entropy_loss(self):
        """
        Compute entropy loss for concept embeddings to encourage diversity.
        """
        p = F.softmax(self.memory_concepts, dim=0)  # Normalize embeddings
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1).mean()  # Mean entropy across concepts
        return -entropy  # Maximize entropy by minimizing negative entropy

    def completeness_loss(self, memory_output, text_embedding, image_embedding):
        """
        Compute completeness loss: difference between memory output and concatenated input embeddings.
        """
        # **Concatenate text and image embeddings** (Shape: [batch, 1536])
        input_representation = torch.cat((text_embedding, image_embedding), dim=-1)

        # **Compute L2 difference**
        loss = torch.norm(memory_output - input_representation, p=2, dim=-1).mean()
        return loss  # Returns a scalar loss value

    def encode_text(self, input_ids, attention_mask):
        """Encode text input using CLIP"""
        with torch.no_grad():
            text_embedding = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return text_embedding  # Shape: [batch, 768]

    def encode_image(self, pixel_values):
        """Encode image input using CLIP"""
        with torch.no_grad():
            image_embedding = self.clip_model.get_image_features(pixel_values=pixel_values)
        return image_embedding  # Shape: [batch, 768]

    def attention_memory_lookup(self, text_embedding, image_embedding):
        """
        Compute weighted sum of memory using soft attention:
        - Computes separate attention for text and image.
        - Uses only relevant parts of memory for each.
        - Concatenates the weighted memory outputs.
        """
        # **Split Memory into Two Parts**
#         text_memory = self.memory_concepts[:, :768]  # First half for text
#         image_memory = self.memory_concepts[:, 768:]  # Second half for image
        text_memory = self.memory_concepts  # First half for text
        image_memory = self.memory_concepts

        # **Compute Attention Scores Separately**
        text_attention_scores = torch.matmul(text_embedding, text_memory.T)  # [batch, memory_size]
        image_attention_scores = torch.matmul(image_embedding, image_memory.T)  # [batch, memory_size]

        # **Apply Softmax Separately**
        text_attention_weights = F.softmax(text_attention_scores, dim=-1)  # [batch, memory_size]
        image_attention_weights = F.softmax(image_attention_scores, dim=-1)  # [batch, memory_size]

        # **Compute Separate Weighted Sums**
        text_memory_output = torch.matmul(text_attention_weights, text_memory)  # [batch, 768]
        image_memory_output = torch.matmul(image_attention_weights, image_memory)  # [batch, 768]

        # **Concatenate Both Outputs**
        memory_output = torch.cat((text_memory_output, image_memory_output), dim=-1)  # [batch, 1536]

        return memory_output, text_attention_weights, image_attention_weights

    def forward(self, text_input_ids=None, text_attention_mask=None, image_pixel_values=None):
        """
        Forward pass:
        - Encode input (text & image)
        - Compute separate memory attention
        - Concatenate memory outputs before classification
        """
        if text_input_ids is not None and text_attention_mask is not None:
            text_embedding = self.encode_text(text_input_ids, text_attention_mask)  # [batch, 768]
        else:
            text_embedding = torch.zeros((image_pixel_values.shape[0], self.embedding_dim), device=self.device)

        if image_pixel_values is not None:
            image_embedding = self.encode_image(image_pixel_values)  # [batch, 768]
        else:
            image_embedding = torch.zeros((text_input_ids.shape[0], self.embedding_dim), device=self.device)

        # **Compute Memory Attention Separately for Text & Image**
        memory_output, text_attention_weights, image_attention_weights = self.attention_memory_lookup(text_embedding, image_embedding)

        # **Final Classification Using MLP**
        logits = self.mlp(memory_output)

       #return logits, text_attention_weights, image_attention_weights
        return logits, memory_output, text_embedding, image_embedding, text_attention_weights, image_attention_weights
