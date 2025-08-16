import torch
import torch.nn.functional as F
from model import StudentModel, get_dinov2_model, FeatureAdapter
from loss import HeterogeneousKnowledgeDistillationLoss

def test_model_forward():
    """Test forward pass of models with sequences"""
    # Create dummy input - sequence of images
    batch_size = 2
    sequence_length = 10  # Smaller sequence for testing
    dummy_input = torch.randn(batch_size, sequence_length, 3, 224, 224)
    
    # Test student model
    student_model = StudentModel(num_classes=3, sequence_length=sequence_length, embed_dim=768)
    student_logits, student_features = student_model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Student model output shape: {student_logits.shape}")
    print(f"Student features shape: {student_features.shape}")
    
    # Test teacher model with sequences
    teacher_model = get_dinov2_model('dinov2_vitb14')
    if teacher_model is not None:
        teacher_model.eval()
        with torch.no_grad():
            # Process sequences through teacher (flatten first)
            sequences_flat = dummy_input.view(batch_size * sequence_length, *dummy_input.shape[2:])
            
            # Get intermediate layers from teacher
            intermediate_layers = teacher_model.get_intermediate_layers(sequences_flat, n=1)
            teacher_features_flat = intermediate_layers[0][:, 1:]  # Remove cls token
            
            # Reshape back to sequence and average
            num_patches, feature_dim = teacher_features_flat.shape[1], teacher_features_flat.shape[2]
            teacher_features_seq = teacher_features_flat.view(batch_size, sequence_length, num_patches, feature_dim)
            teacher_features = torch.mean(teacher_features_seq, dim=1)  # Average across sequence
            
            # Resize teacher features to match student features if needed
            if teacher_features.shape[1] != student_features.shape[1]:
                # Use interpolation to match dimensions
                # teacher_features: [B, N_t, D] -> [B, N_s, D]
                teacher_features = F.interpolate(
                    teacher_features.permute(0, 2, 1), 
                    size=student_features.shape[1], 
                    mode='linear', 
                    align_corners=False
                ).permute(0, 2, 1)
            
        print(f"Teacher features shape: {teacher_features.shape}")
        
        # Test feature adapter
        adapter = FeatureAdapter(768, 768)  # Student dim to teacher dim
        student_features_adapted = adapter(student_features)
        print(f"Adapted student features shape: {student_features_adapted.shape}")
        
        # Test loss function
        labels = torch.randint(0, 3, (batch_size,))
        criterion = HeterogeneousKnowledgeDistillationLoss()
        
        total_loss, loss_components = criterion(
            student_logits, teacher_features, student_features, 
            labels, adapter
        )
        
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Task loss: {loss_components['task_loss'].item():.4f}")
        print(f"Feature loss: {loss_components['feature_loss'].item():.4f}")
        print(f"Relation loss: {loss_components['relation_loss'].item():.4f}")
        
        print("All sequence model tests passed!")
    else:
        print("Failed to load teacher model - using offline mode")
        print("Student model sequence processing works correctly!")

if __name__ == "__main__":
    test_model_forward()