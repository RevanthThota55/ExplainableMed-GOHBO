# Quick Start Guide - New Features

## 🚀 Getting Started with Explainability & Deployment

### ⚡ 5-Minute Setup

#### 1. Update Your Files (REQUIRED)
```bash
cd "D:/Major Project/medical-image-classification"

# Backup and replace ResNet-18 model
mv src/models/resnet18_medical.py src/models/resnet18_medical_backup.py
mv src/models/resnet18_medical_updated.py src/models/resnet18_medical.py

# Backup and replace requirements
mv requirements.txt requirements_backup.txt
mv requirements_updated.txt requirements.txt
```

#### 2. Install New Dependencies
```bash
pip install onnx>=1.15.0 onnxruntime>=1.16.0
```

#### 3. Verify Installation
```python
python -c "
import onnx
import onnxruntime
print('✅ ONNX and ONNX Runtime installed successfully!')
"
```

---

## 🎨 Feature 1: Grad-CAM (Visual Explanations)

### Basic Usage
```python
from src.explainability.gradcam import GradCAM
from src.models.resnet18_medical import MedicalResNet18
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load your trained model
model = MedicalResNet18(num_classes=4)
model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])
model.eval()

# Initialize Grad-CAM
gradcam = GradCAM(model, target_layer='layer4', device='cuda')

# Load and preprocess image
image = Image.open('path/to/medical_image.jpg')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(image).unsqueeze(0)

# Generate heatmap
heatmap = gradcam.generate_heatmap(input_tensor, class_idx=1)  # or None for predicted class

# Overlay heatmap on original image
import numpy as np
original_np = np.array(image.resize((224, 224)))
overlay = gradcam.overlay_heatmap(original_np, heatmap, alpha=0.4)

# Display or save
import matplotlib.pyplot as plt
plt.imshow(overlay)
plt.title('Grad-CAM Visualization')
plt.axis('off')
plt.savefig('gradcam_result.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Batch Visualization
```python
from src.explainability.gradcam import create_gradcam_grid

# Assuming you have a dataloader
images, labels = next(iter(test_loader))

# Convert to original images (denormalize)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
originals = images * std + mean
originals = [img.permute(1, 2, 0).numpy() for img in originals]

# Create grid
create_gradcam_grid(
    model, images, originals,
    class_names=['glioma', 'meningioma', 'no_tumor', 'pituitary'],
    true_labels=labels.tolist(),
    grid_size=(2, 2),
    save_path='gradcam_grid.png'
)
```

---

## 🎲 Feature 2: MC Dropout (Uncertainty Estimation)

### Basic Usage
```python
from src.explainability.uncertainty import MCDropoutPredictor
from src.models.resnet18_medical import MedicalResNet18

# Load model with MC Dropout enabled
model = MedicalResNet18(num_classes=4, enable_mc_dropout=True)
model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])

# Initialize predictor
mc_predictor = MCDropoutPredictor(model, num_passes=20, device='cuda')

# Predict with uncertainty
image = preprocess(Image.open('path/to/image.jpg')).unsqueeze(0)
mean_prediction, uncertainty = mc_predictor.predict_with_uncertainty(image)

# Print results
print(f"Predicted class: {uncertainty['predicted_class']}")
print(f"Confidence: {uncertainty['confidence']:.2%}")
print(f"Entropy: {uncertainty['entropy']:.4f}")
print(f"Mean Std Dev: {uncertainty['mean_std']:.4f}")

# Flag uncertain predictions
if uncertainty['confidence'] < 0.85:
    print("⚠️  UNCERTAIN - Recommend human review")
```

### Dataset Analysis
```python
from src.explainability.uncertainty import UncertaintyAnalyzer

# Initialize analyzer
analyzer = UncertaintyAnalyzer(
    mc_predictor,
    class_names=['glioma', 'meningioma', 'no_tumor', 'pituitary'],
    confidence_threshold=0.85
)

# Analyze entire test set
results = analyzer.analyze_dataset(test_loader, max_samples=100)

# Print statistics
print(f"\nTotal samples: {results['statistics']['total_samples']}")
print(f"Uncertain: {results['statistics']['num_uncertain']} ({results['statistics']['uncertainty_rate']:.1%})")
print(f"Mean confidence: {results['statistics']['mean_confidence']:.4f}")
print(f"Accuracy: {results['statistics']['accuracy']:.2%}")

# Plot distributions
analyzer.plot_uncertainty_distribution(results, save_path='uncertainty_distribution.png')

# Generate report
report = analyzer.generate_uncertainty_report(results, save_path='uncertainty_report.txt')
print(report)

# Save uncertain cases for review
analyzer.save_uncertain_cases(results, Path('uncertain_cases.json'))
```

---

## 📦 Feature 3: Model Quantization

### Quantize a Trained Model
```python
from src.deployment.quantize import ModelQuantizer
from torch.utils.data import DataLoader, Subset

# Load trained model
model = MedicalResNet18(num_classes=4)
model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])

# Create calibration dataset (100 samples from training set)
quantizer = ModelQuantizer(model)
calib_loader = quantizer.prepare_calibration_data(train_loader, num_samples=100)

# Quantize model
quantized_model = quantizer.quantize_static(backend='fbgemm')

# Validate
results = quantizer.validate_quantized_model(test_loader)

# Expected output:
# ✓ Accuracy loss: <2%
# ✓ Speedup: 2-4x
# ✓ Size reduction: ~75%

# Save quantized model
quantizer.save_quantized_model(Path('models/quantized_model.pth'))
```

### Use Quantized Model for Inference
```python
# Load quantized model
quantized_model = torch.load('models/quantized_model_full.pth')
quantized_model.eval()

# Standard inference (much faster on CPU!)
with torch.no_grad():
    output = quantized_model(image)
    prediction = output.argmax(dim=1)
```

---

## 🌐 Feature 4: ONNX Export

### Export Model to ONNX
```python
from src.deployment.export_onnx import ONNXExporter
from pathlib import Path

# Load trained model
model = MedicalResNet18(num_classes=4)
model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])

# Initialize exporter
exporter = ONNXExporter(model, input_shape=(1, 3, 224, 224))

# Export to ONNX
onnx_path = exporter.export(
    save_path=Path('models/medical_model.onnx'),
    opset_version=14
)

# Verify export
verification = exporter.verify_export()
print(f"Verified: {verification['verified']}")
print(f"Max difference: {verification['max_difference']:.2e}")

# The exporter automatically creates inference example code:
# models/medical_model_inference.py
```

### Use ONNX Model for Inference
```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('models/medical_model.onnx')

# Prepare input
image_np = preprocess(image).unsqueeze(0).numpy()

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_np})

# Get prediction
probabilities = output[0]
predicted_class = np.argmax(probabilities)
confidence = probabilities[0][predicted_class]

print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
```

### Benchmark ONNX Model
```python
# Prepare test inputs
test_inputs = [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(100)]

# Benchmark
benchmark_results = exporter.benchmark_onnx(test_inputs, warmup_runs=10, num_runs=100)

print(f"Mean latency: {benchmark_results['mean_latency_ms']:.2f} ms")
print(f"P95 latency: {benchmark_results['p95_latency_ms']:.2f} ms")
print(f"Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
```

---

## 🔄 Complete Workflow Example

### Train → Explain → Quantize → Export

```python
"""
Complete workflow: From training to deployment with explainability
"""

# 1. Train model (existing workflow)
# python train.py --dataset brain_tumor --learning_rate optimized --epochs 100

# 2. Load trained model
from src.models.resnet18_medical import MedicalResNet18
model = MedicalResNet18(num_classes=4, enable_mc_dropout=True)
model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])

# 3. Generate explanations
from src.explainability.gradcam import GradCAM
from src.explainability.uncertainty import MCDropoutPredictor, UncertaintyAnalyzer

gradcam = GradCAM(model)
mc_predictor = MCDropoutPredictor(model, num_passes=20)
analyzer = UncertaintyAnalyzer(mc_predictor, class_names)

# Analyze uncertainty on test set
uncertainty_results = analyzer.analyze_dataset(test_loader)
analyzer.generate_uncertainty_report(uncertainty_results, Path('reports/uncertainty.txt'))
analyzer.save_uncertain_cases(uncertainty_results, Path('reports/uncertain_cases.json'))

# 4. Generate Grad-CAM visualizations for uncertain cases
uncertain_indices = uncertainty_results['uncertain_indices'][:10]  # Top 10 uncertain
for idx in uncertain_indices:
    image, label = test_dataset[idx]
    heatmap = gradcam.generate_heatmap(image.unsqueeze(0))
    # Save visualization...

# 5. Quantize for deployment
from src.deployment.quantize import quantize_model
quantized_model, quant_results = quantize_model(
    model, calib_loader, test_loader,
    save_path=Path('models/quantized_model.pth')
)

# 6. Export to ONNX
from src.deployment.export_onnx import export_and_verify
onnx_path, onnx_results = export_and_verify(
    quantized_model,  # Can export quantized model!
    Path('models/medical_model_quantized.onnx')
)

print("✅ Complete pipeline finished!")
print(f"   - Uncertainty analysis: {len(uncertainty_results['uncertain_indices'])} uncertain cases flagged")
print(f"   - Quantized model: {quant_results['size_reduction_pct']:.1f}% smaller, {quant_results['speedup']:.2f}x faster")
print(f"   - ONNX model: Verified and ready for deployment")
```

---

## 📱 Deployment Examples

### Desktop Application (CPU)
```python
# Use quantized model for fast CPU inference
quantized_model = torch.load('models/quantized_model_full.pth')
mc_predictor = MCDropoutPredictor(quantized_model)

# Fast inference with uncertainty
mean_pred, uncertainty = mc_predictor.predict_with_uncertainty(image)
if uncertainty['confidence'] > 0.90:
    return mean_pred  # High confidence
else:
    return "Uncertain - consult specialist"
```

### Mobile/Edge Device (ONNX)
```python
# Use ONNX Runtime (works on iOS, Android, Jetson, etc.)
import onnxruntime as ort
session = ort.InferenceSession('medical_model.onnx')
output = session.run(None, {input_name: image_np})
```

### Web Application (ONNX.js)
```javascript
// JavaScript inference using ONNX.js
const session = await ort.InferenceSession.create('medical_model.onnx');
const results = await session.run({input: imageTensor});
```

---

## 🎓 Learning Resources

### Understand the Outputs

**Grad-CAM Heatmap Colors:**
- 🔴 Red: High activation (important for prediction)
- 🟡 Yellow: Moderate activation
- 🟢 Green: Low activation
- 🔵 Blue: No activation

**Uncertainty Metrics:**
- **Confidence** (0-1): Higher = more certain (> 0.85 recommended)
- **Entropy** (0 to log(classes)): Lower = more certain
- **Std Dev** (0-1): Lower = more consistent predictions

**Quantization Results:**
- **Accuracy loss < 1%**: Excellent quantization
- **Accuracy loss 1-2%**: Good quantization
- **Accuracy loss > 2%**: May need more calibration samples

---

## ⚡ Pro Tips

1. **Grad-CAM**: Use alpha=0.3-0.5 for best visualization clarity
2. **MC Dropout**: 20 passes is good balance between accuracy and speed (10 for faster, 50 for better)
3. **Quantization**: Use validation set for calibration if accuracy loss is high
4. **ONNX**: Test on target device - performance varies by hardware

---

## 🐛 Troubleshooting

**Q: Grad-CAM shows random patterns**
A: Ensure model is loaded correctly and in eval mode

**Q: MC Dropout predictions don't vary**
A: Check `enable_mc_dropout=True` and dropout_rate > 0

**Q: Quantization fails**
A: Model must be on CPU, try `model.cpu()` first

**Q: ONNX export error**
A: Check for custom layers, may need to modify model

---

## ✅ Next Steps

1. ✅ Test each feature individually
2. ✅ Generate explanations for your trained models
3. ✅ Quantize and benchmark on target hardware
4. ✅ Export to ONNX for deployment
5. ✅ Create clinical report with Grad-CAM visualizations

**Congratulations! Your project now has production-ready explainability and deployment capabilities!** 🎉

---

For detailed documentation, see `IMPLEMENTATION_SUMMARY.md`