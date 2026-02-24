"""

     TEST SYSTEM - Valider l'installation et la configuration  
  Vérifier PyTorch, CUDA, dépendances et fichiers               


Usage:
  python test_system.py
"""

import sys
import os
from pathlib import Path

print("\n" + "="*70)
print("   SYSTEM VALIDATION - TÂCHE 2")
print("="*70 + "\n")

# 
# TEST 1: Version Python
# 

print("1⃣  Python Version:")
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"   Python {py_version}")

if sys.version_info >= (3, 9):
    print("    OK (3.9+ required)\n")
else:
    print(f"    FAILED (3.9+ required, got {py_version})\n")
    sys.exit(1)


# 
# TEST 2: PyTorch
# 

print("2⃣  PyTorch:")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"    Imported successfully\n")
except ImportError as e:
    print(f"    Failed to import PyTorch: {e}")
    print("   Run: pip install torch\n")
    sys.exit(1)


# 
# TEST 3: CUDA/GPU
# 

print("3⃣  CUDA / GPU:")
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"    CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Device Count: {torch.cuda.device_count()}\n")
else:
    print(f"     CUDA not available (using CPU)")
    print(f"   Note: CPU mode is slower but works fine\n")


# 
# TEST 4: PyTorch Forward Pass
# 

print("4⃣  PyTorch Forward Pass:")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(1, 4, 224, 224, device=device)
    print(f"   Created test tensor: {x.shape}")
    print(f"    OK\n")
except Exception as e:
    print(f"    Failed: {e}\n")
    sys.exit(1)


# 
# TEST 5: Core Dependencies
# 

print("5⃣  Core Dependencies:")
dependencies = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'sklearn': 'scikit-learn',
    'tqdm': 'tqdm',
    'PIL': 'Pillow',
}

all_ok = True
for module_name, display_name in dependencies.items():
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"    {display_name:20} {version}")
    except ImportError:
        print(f"    {display_name:20} NOT INSTALLED")
        all_ok = False

print()
if not all_ok:
    print("   Run: pip install -r requirements.txt\n")
    sys.exit(1)


# 
# TEST 6: Model Architecture
# 

print("6⃣  Model Architecture:")
try:
    from map_width_regressor import MapWidthRegressor, CONFIG
    
    model = MapWidthRegressor(num_channels=4, num_output=1)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model: MapWidthRegressor")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    
    # Forward pass test
    test_input = torch.randn(2, 4, 224, 224, device=device)
    with torch.no_grad():
        test_output = model(test_input)
    
    print(f"   Input Shape: {test_input.shape}")
    print(f"   Output Shape: {test_output.shape}")
    print(f"    OK\n")
    
except Exception as e:
    print(f"    Failed: {e}\n")
    sys.exit(1)


# 
# TEST 7: Directory Structure
# 

print("7⃣  Directory Structure:")
task_dir = Path(__file__).parent
data_dir = task_dir / 'data'
outputs_dir = task_dir / 'outputs'

print(f"   Task directory: {task_dir}")

if data_dir.exists():
    npz_count = len(list(data_dir.glob('**/*.npz')))
    print(f"    data/ exists ({npz_count} .npz files)")
else:
    print(f"     data/ not found (create it and add .npz files)")

if outputs_dir.exists():
    print(f"    outputs/ exists")
else:
    print(f"     outputs/ not found (will be created during training)")

print()


# 
# TEST 8: Dataset Loading (if files exist)
# 

print("8⃣  Dataset Loading:")
try:
    npz_files = list(data_dir.glob('**/*.npz'))
    
    if len(npz_files) == 0:
        print(f"     No .npz files found in data/")
        print(f"      (This is OK for first-time setup)\n")
    else:
        # Try loading first file
        sample_file = npz_files[0]
        npz_data = np.load(sample_file, allow_pickle=True)
        
        print(f"   Sample file: {sample_file.name}")
        print(f"   Keys: {list(npz_data.files)}")
        
        if 'data' in npz_data.files:
            mag_shape = npz_data['data'].shape
            print(f"   Magnetic data shape: {mag_shape}")
            
            if mag_shape[2] == 4:
                print(f"    Correct 4-channel format")
            else:
                print(f"    Expected 4 channels, got {mag_shape[2]}")
        
        if 'width' in npz_data.files or 'label' in npz_data.files:
            width_val = npz_data.get('width', npz_data.get('label'))
            print(f"   Label (width): {width_val} meters")
            print(f"    Label found")
        else:
            print(f"    No label ('width' or 'label') found")
        
        print()

except Exception as e:
    print(f"     Could not load sample: {e}\n")


# 
# SUMMARY
# 

print("="*70)
print("   SYSTEM VALIDATION COMPLETE")
print("="*70)
print()

print(" Configuration Summary:")
print(f"   • Python: {py_version}")
print(f"   • PyTorch: {torch.__version__}")
print(f"   • Device: {'GPU (CUDA)' if cuda_available else 'CPU'}")
print(f"   • Model: MapWidthRegressor ({total_params:,} params)")
print()

print(" Next Steps:")
print("   1. If data/ is empty: Add your .npz files")
print("   2. Run: python analyze_width_dataset.py")
print("   3. Run: python map_width_regressor.py")
print()

print(" For more info: README_TASK2.md\n")
