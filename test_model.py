import torch
import torch.nn.functional as F
from model import StudentModel, get_dinov2_model, FeatureAdapter
from loss import HeterogeneousKnowledgeDistillationLoss

def test_model_forward():
    """Test forward pass of models"""
    # Create dummy input - using smaller size for faster testing
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Test student model
    student_model = StudentModel(num_classes=3)
    student_logits, student_features = student_model(dummy_input)
    
    print(f"Student model output shape: {student_logits.shape}")
    print(f"Student features shape: {student_features.shape}")
    
    # Test teacher model
    teacher_model = get_dinov2_model('dinov2_vits14')
    if teacher_model is not None:
        teacher_model.eval()
        with torch.no_grad():
            # Get intermediate layers from teacher
            intermediate_layers = teacher_model.get_intermediate_layers(dummy_input, n=1)
            teacher_features = intermediate_layers[0][:, 1:]  # Remove cls token
            
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
        adapter = FeatureAdapter(384, 384)  # Student dim to teacher dim
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
        
        print("All tests passed!")
    else:
        print("Failed to load teacher model")

if __name__ == "__main__":
    test_model_forward()