import json
from pathlib import Path
from torchvision import transforms

BASE_DIR     = Path(__file__).parent.parent
CKPT_PATH    = BASE_DIR / 'checkpoints' / 'best_densenet_v2.pth'
RESULTS_PATH = BASE_DIR / 'Notebooks'   / 'ta005f_final_results.json'
PARAMS_PATH  = BASE_DIR / 'Notebooks'   / 'best_params_densenet.json'

CLASSES     = ['alveolar_pattern', 'bronchial_pattern', 'pleural_effusion', 'cardiomegaly', 'no_finding']
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224

ALLOWED_EXTENSIONS = {'.dcm', '.dicom', '.png', '.jpg', '.jpeg'}

with open(RESULTS_PATH) as f:
    _results = json.load(f)
THRESHOLDS = _results['thresholds']

with open(PARAMS_PATH) as f:
    _params = json.load(f)
DROPOUT = _params['params']['dropout']

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
