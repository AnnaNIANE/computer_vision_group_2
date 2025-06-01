from ultralytics import YOLO

# Charger le modèle pré-entraîné
model = YOLO('yolov8n.pt')  # Utilise yolov8n.pt pour un modèle plus léger

# Fine-tuner le modèle avec ton dataset
model.train(
    data='/home/students-ans29/Bureau/cv/Project_computer_vision1/Project_computer_vision/PH Ambulances.v1i.yolov8/data.yaml',
    epochs=15,  # 30 époques pour un entraînement rapide
    imgsz=640,  # Taille des images (standard pour Roboflow)
    batch=4,    # Taille du batch (ajustée pour un CPU, augmente à 16 si GPU disponible)
    project='/home/students-ans29/Bureau/cv/Project_computer_vision1/Project_computer_vision/runs/train',  # Dossier pour sauvegarder les résultats
    name='fine_tune_ambulance',  # Nom de l'entraînement
    patience=10,  # Arrête si aucune amélioration après 10 époques
    device='cpu'  # Utilise le CPU (change à 0 si GPU disponible)
)

# Sauvegarder le modèle fine-tuné
model.save('/home/students-ans29/Bureau/cv/Project_computer_vision1/Project_computer_vision/yolov8n_finetuned.pt')