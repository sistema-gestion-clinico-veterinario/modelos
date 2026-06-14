import time
import torch
import torch.nn as nn
from torchvision import models
from config import CLASSES, NUM_CLASSES, DROPOUT, THRESHOLDS, CKPT_PATH, TRANSFORM
from preprocessing import prepare


class ModelPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = None

    def load(self):
        m = models.densenet121(weights=None, memory_efficient=True)
        m.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(m.classifier.in_features, NUM_CLASSES)
        )
        m.load_state_dict(torch.load(CKPT_PATH, map_location=self.device))
        m.eval()
        self.model = m.to(self.device)
        print(f'[predictor] Modelo cargado — device: {self.device}')

    def predict(self, file_bytes: bytes, filename: str) -> dict:
        if self.model is None:
            raise RuntimeError('Modelo no cargado. Llama a load() primero.')

        t0 = time.time()
        img, file_type = prepare(file_bytes, filename)
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.sigmoid(self.model(tensor))[0].cpu().numpy()

        predictions = {}
        diagnoses   = []
        for i, cls in enumerate(CLASSES):
            thr      = THRESHOLDS.get(cls, 0.5)
            prob     = float(probs[i])
            positive = prob >= thr
            predictions[cls] = {
                'probability': round(prob, 4),
                'positive':    positive,
                'threshold':   thr
            }
            if positive:
                diagnoses.append(cls)

        return {
            'model':        'densenet121',
            'file_type':    file_type,
            'predictions':  predictions,
            'diagnoses':    diagnoses,
            'inference_ms': round((time.time() - t0) * 1000, 2)
        }


predictor = ModelPredictor()
