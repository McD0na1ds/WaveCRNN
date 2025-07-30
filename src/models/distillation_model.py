import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy, ConfusionMatrix
from transformers import Dinov2Model, Dinov2Config
import hydra
from omegaconf import DictConfig
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from PIL import Image

from .student_model import StudentCRNN

logger = logging.getLogger(__name__)


class TeacherModel(nn.Module):
    def __init__(self, model_name="dinov2-base", num_classes=3):
        super().__init__()
        
        try:
            # Load pretrained DINOv2
            self.dinov2 = Dinov2Model.from_pretrained(f"facebook/{model_name}")
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
        
        # Freeze the model
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        # Add classification head
        hidden_size = self.dinov2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x, return_features=False):
        with torch.no_grad():
            outputs = self.dinov2(x)
            features = outputs.last_hidden_state[:, 0]  # CLS token
            
        if return_features:
            return features
            
        logits = self.classifier(features)
        return logits


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, targets, classification_loss):
        # Distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * classification_loss
        
        return total_loss, distill_loss


class FeatureDistillationLoss(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_features, teacher_features):
        # L2 normalization
        student_features = F.normalize(student_features, p=2, dim=1)
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
        
        # MSE loss between normalized features
        feature_loss = self.mse_loss(student_features, teacher_features)
        return feature_loss


class DistillationLightningModule(L.LightningModule):
    def __init__(
        self, 
        teacher, 
        student, 
        loss_weights, 
        learning_rate, 
        weight_decay, 
        temperature, 
        num_classes=3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Models
        self.teacher = TeacherModel(teacher['model_name'], num_classes)
        if teacher['freeze']:
            self.teacher.eval()
            
        self.student = StudentCRNN(
            vit_config=student['vit'],
            lstm_config=student['lstm'],
            num_classes=num_classes
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.distillation_loss = DistillationLoss(temperature=temperature, alpha=0.5)
        self.feature_distillation_loss = FeatureDistillationLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        if num_classes >= 5:
            self.train_acc_top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
            self.val_acc_top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
            self.test_acc_top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        
        # Confusion Matrix
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.num_classes = num_classes
        
        # Class names for visualization
        self.class_names = ["Plunging", "Spilling", "Surging"]
        
    def forward(self, x):
        return self.student(x)
    
    def _shared_step(self, batch, batch_idx, stage):
        images, targets = batch
        
        # Get teacher predictions and features
        teacher_logits = self.teacher(images)
        teacher_features = self.teacher(images, return_features=True)
        
        # Get student predictions and features
        student_features, student_logits = self.student.get_features_and_logits(images)
        
        # Calculate losses
        classification_loss = self.classification_loss(student_logits, targets)
        
        # Knowledge distillation loss
        distill_loss, kl_loss = self.distillation_loss(
            student_logits, teacher_logits, targets, classification_loss
        )
        
        # Feature distillation loss
        feature_loss = self.feature_distillation_loss(student_features, teacher_features)
        
        # Combined loss
        total_loss = (
            self.loss_weights['classification'] * classification_loss +
            self.loss_weights['distillation'] * kl_loss +
            0.1 * feature_loss  # Feature distillation weight
        )
        
        # Predictions
        preds = torch.argmax(student_logits, dim=1)
        
        # Log losses
        self.log(f"{stage}/total_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/classification_loss", classification_loss, sync_dist=True)
        self.log(f"{stage}/distillation_loss", kl_loss, sync_dist=True)
        self.log(f"{stage}/feature_loss", feature_loss, sync_dist=True)
        
        return {
            'loss': total_loss,
            'preds': preds,
            'targets': targets,
            'student_logits': student_logits,
            'teacher_logits': teacher_logits
        }
    
    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, batch_idx, "train")
        
        # Update metrics
        self.train_acc(outputs['preds'], outputs['targets'])
        if self.num_classes >= 5:
            self.train_acc_top5(outputs['student_logits'], outputs['targets'])
        
        # Log metrics
        self.log("train/accuracy", self.train_acc, prog_bar=True, on_epoch=True, sync_dist=True)
        if self.num_classes >= 5:
            self.log("train/accuracy_top5", self.train_acc_top5, on_epoch=True, sync_dist=True)
        
        return outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, batch_idx, "val")
        
        # Update metrics
        self.val_acc(outputs['preds'], outputs['targets'])
        if self.num_classes >= 5:
            self.val_acc_top5(outputs['student_logits'], outputs['targets'])
        
        # Update confusion matrix
        self.val_confusion_matrix(outputs['preds'], outputs['targets'])
        
        return outputs
    
    def on_validation_epoch_end(self):
        # Log metrics
        self.log("val/accuracy", self.val_acc.compute(), prog_bar=True, sync_dist=True)
        if self.num_classes >= 5:
            self.log("val/accuracy_top5", self.val_acc_top5.compute(), sync_dist=True)
        
        # Log confusion matrix
        if self.trainer.is_global_zero:  # Only log on main process
            self._log_confusion_matrix(self.val_confusion_matrix, "val")
        
        # Reset confusion matrix
        self.val_confusion_matrix.reset()
    
    def test_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, batch_idx, "test")
        
        # Update metrics
        self.test_acc(outputs['preds'], outputs['targets'])
        if self.num_classes >= 5:
            self.test_acc_top5(outputs['student_logits'], outputs['targets'])
        
        # Update confusion matrix
        self.test_confusion_matrix(outputs['preds'], outputs['targets'])
        
        return outputs
    
    def on_test_epoch_end(self):
        # Log metrics
        self.log("test/accuracy", self.test_acc.compute(), sync_dist=True)
        if self.num_classes >= 5:
            self.log("test/accuracy_top5", self.test_acc_top5.compute(), sync_dist=True)
        
        # Log confusion matrix
        if self.trainer.is_global_zero:
            self._log_confusion_matrix(self.test_confusion_matrix, "test")
    
    def _log_confusion_matrix(self, confusion_matrix, stage):
        """Log confusion matrix to wandb"""
        try:
            cm = confusion_matrix.compute().cpu().numpy()
            
            # Create confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.title(f'{stage.capitalize()} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            
            # Log to wandb
            if hasattr(self.logger, 'experiment'):
                import wandb
                self.logger.experiment.log({
                    f"{stage}/confusion_matrix": wandb.Image(img),
                    "global_step": self.global_step
                })
            
            plt.close()
            buf.close()
            
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")
    
    def configure_optimizers(self):
        # Only optimize student parameters
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warmup
        def lr_lambda(current_step):
            if current_step < 1000:  # Warmup
                return float(current_step) / float(max(1, 1000))
            else:
                # Cosine annealing
                progress = float(current_step - 1000) / float(max(1, self.trainer.estimated_stepping_batches - 1000))
                return max(0.01, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def on_train_start(self):
        """Log model architecture"""
        if self.trainer.is_global_zero:
            logger.info(f"Teacher model: {self.teacher}")
            logger.info(f"Student model: {self.student}")
            
            # Count parameters
            teacher_params = sum(p.numel() for p in self.teacher.parameters())
            student_params = sum(p.numel() for p in self.student.parameters())
            trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            
            logger.info(f"Teacher parameters: {teacher_params:,}")
            logger.info(f"Student parameters: {student_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            # Log to wandb
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.config.update({
                    "teacher_params": teacher_params,
                    "student_params": student_params,
                    "trainable_params": trainable_params,
                    "compression_ratio": teacher_params / student_params
                })