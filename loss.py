import torch
import torch.nn as nn
import torch.nn.functional as F

class HeterogeneousKnowledgeDistillationLoss(nn.Module):
    """Composite loss for heterogeneous knowledge distillation"""
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, temperature=4.0):
        """
        Args:
            alpha: Weight for task loss
            beta: Weight for feature imitation loss
            gamma: Weight for relational knowledge loss
            temperature: Temperature for softening probability distributions
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.task_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_features, student_features, 
                labels, feature_adapter):
        """
        Compute the total loss for heterogeneous knowledge distillation
        
        Args:
            student_logits: Output logits from student model
            teacher_features: Features from teacher model
            student_features: Features from student model
            labels: Ground truth labels
            feature_adapter: Adapter to map student features to teacher space
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary with individual loss components
        """
        # Task loss (supervised learning)
        loss_task = self.task_loss(student_logits, labels)
        
        # Feature imitation loss with adapter
        student_features_adapted = feature_adapter(student_features)
        loss_feature = F.mse_loss(student_features_adapted, teacher_features)
        
        # Relational knowledge distillation (Gram matrix loss)
        # Compute normalized Gram matrices
        gram_teacher = self.compute_normalized_gram_matrix(teacher_features)
        gram_student = self.compute_normalized_gram_matrix(student_features)
        
        # Compute relational loss
        loss_relation = F.mse_loss(gram_student, gram_teacher)
        
        # Total loss
        total_loss = (self.alpha * loss_task + 
                     self.beta * loss_feature + 
                     self.gamma * loss_relation)
        
        # Return components for logging
        loss_components = {
            'total_loss': total_loss,
            'task_loss': loss_task,
            'feature_loss': loss_feature,
            'relation_loss': loss_relation
        }
        
        return total_loss, loss_components
    
    def compute_normalized_gram_matrix(self, features):
        """
        Compute normalized Gram matrix for feature relational knowledge
        Args:
            features: Tensor of shape (batch_size, num_patches, feature_dim)
        Returns:
            gram_matrix: Tensor of shape (batch_size, num_patches, num_patches)
        """
        # features: (B, N, D)
        batch_size, num_patches, feature_dim = features.shape
        
        # Compute Gram matrix
        gram_matrix = torch.bmm(features, features.transpose(1, 2))  # (B, N, N)
        
        # Normalize by feature dimension
        gram_matrix = gram_matrix / feature_dim
        
        return gram_matrix