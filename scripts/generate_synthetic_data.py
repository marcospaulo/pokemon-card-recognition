import os
import random
import math
import shutil
import yaml
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import zipfile

def create_directory_structure():
    """Crea la struttura delle directory necessarie per il dataset YOLO."""
    # Directory principali
    base_dirs = ['dataset', 'dataset/images', 'dataset/labels', 
                'dataset/images/train', 'dataset/images/val', 'dataset/images/test',
                'dataset/labels/train', 'dataset/labels/val', 'dataset/labels/test']
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        
    print("✓ Struttura delle directory creata con successo")

def get_card_files(cards_dir):
    """Ottiene la lista dei file delle carte Pokemon."""
    return [os.path.join(cards_dir, f) for f in os.listdir(cards_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

def get_background_files(backgrounds_dir):
    """Ottiene la lista dei file di sfondo."""
    return [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

def calculate_iou(box1, box2):
    """
    Calcola l'IoU (Intersection over Union) tra due bounding box.
    box1, box2: ogni box è rappresentata da [x1, y1, x2, y2]
    """
    # Coordinate dell'intersezione
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Area dell'intersezione
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Area di ciascun box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Unione
    union_area = box1_area + box2_area - intersection_area
    
    # IoU
    if union_area <= 0:
        return 0
    return intersection_area / union_area

def place_card_on_background(card_img, bg_img, existing_boxes=None, max_attempts=50, scale_factor=None):
    """
    Posiziona una carta Pokemon su uno sfondo, usando un approccio a griglia.
    """
    if existing_boxes is None:
        existing_boxes = []
    
    # Ottieni dimensioni delle immagini
    bg_width, bg_height = bg_img.size
    card_width, card_height = card_img.size
    
    # Usa il fattore di scala fornito o genera uno predefinito
    if scale_factor is None:
        scale_factor = random.uniform(0.08, 0.15)
    
    new_card_width = int(bg_width * scale_factor)
    new_card_height = int(card_height * (new_card_width / card_width))
    
    # Salva le dimensioni originali prima della rotazione
    original_width = new_card_width
    original_height = new_card_height
    
    card_img = card_img.resize((new_card_width, new_card_height), Image.LANCZOS)
    
    # Angolo casuale più limitato per mantenere un aspetto ordinato
    # Riduce l'angolo per carte grandi
    if scale_factor > 0.3:
        angle_deg = random.uniform(-5, 5)  # Angolo minore per carte grandi
    else:
        angle_deg = random.uniform(-15, 15)
        
    card_img = card_img.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
    
    # Ottieni le nuove dimensioni dopo la rotazione
    rotated_width, rotated_height = card_img.size
    
    # Definisci una griglia
    grid_cols = 5  # Numero di colonne nella griglia
    grid_rows = 4  # Numero di righe nella griglia
    cell_width = bg_width // grid_cols
    cell_height = bg_height // grid_rows
    
    # Prova posizioni nella griglia
    for _ in range(max_attempts):
        # Scegli una cella della griglia
        grid_x = random.randint(0, grid_cols - 1)
        grid_y = random.randint(0, grid_rows - 1)
        
        # Calcola la posizione base nella cella con un po' di variazione casuale
        base_x = grid_x * cell_width + random.randint(-10, 10)
        base_y = grid_y * cell_height + random.randint(-10, 10)
        
        # Calcola la posizione effettiva, assicurandosi che sia all'interno dell'immagine
        paste_x = max(0, min(base_x, bg_width - rotated_width))
        paste_y = max(0, min(base_y, bg_height - rotated_height))
        
        # Calcola la bounding box in coordinate assolute (pixel)
        x1 = paste_x
        y1 = paste_y
        x2 = paste_x + rotated_width
        y2 = paste_y + rotated_height
        
        # Controlla se la bounding box è valida
        if x1 >= bg_width or y1 >= bg_height or x2 <= 0 or y2 <= 0:
            continue
        
        # Assicurati che la bounding box sia all'interno dell'immagine
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(bg_width, x2)
        y2 = min(bg_height, y2)
        
        # Controlla se la bounding box è abbastanza grande
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        
        # Crea bbox in formato pixel per controllo sovrapposizioni
        pixel_bbox = [x1, y1, x2, y2]
        
        # Abbassa la soglia IoU per permettere carte più vicine
        overlap_detected = False
        for existing_box in existing_boxes:
            iou = calculate_iou(pixel_bbox, existing_box)
            if iou > 0.1:  # Soglia IoU ridotta al 10%
                overlap_detected = True
                break
                
        if not overlap_detected:
            # Crea una copia dello sfondo per evitare modifiche all'originale
            result_img = bg_img.copy()
            
            # Crea una maschera per la trasparenza
            if card_img.mode == 'RGBA':
                mask = card_img.split()[3]
            else:
                mask = None
                
            # Incolla la carta sullo sfondo
            result_img.paste(card_img, (paste_x, paste_y), mask)
            
            # Calcola i quattro angoli del rettangolo ruotato
            # Per OBB, abbiamo bisogno delle coordinate dei 4 angoli
            # Calcola il centro della carta
            center_x = paste_x + rotated_width / 2
            center_y = paste_y + rotated_height / 2
            
            # Calcola i quattro angoli prima della rotazione (relativo al centro)
            # Usa le dimensioni originali della carta, non quelle dopo la rotazione
            half_width = original_width / 2
            half_height = original_height / 2
            corners = [
                [-half_width, -half_height],  # Top-left
                [half_width, -half_height],   # Top-right
                [half_width, half_height],    # Bottom-right
                [-half_width, half_height]    # Bottom-left
            ]
            
            # Converti l'angolo in radianti (per il calcolo della rotazione)
            angle_rad = math.radians(-angle_deg)  # Negativo perché PIL ruota in senso antiorario
            
            # Applica la rotazione e trasla al centro
            rotated_corners = []
            for x, y in corners:
                # Applica la rotazione
                x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
                y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
                
                # Trasla al centro e normalizza per YOLO
                x_final = (center_x + x_rot) / bg_width
                y_final = (center_y + y_rot) / bg_height
                
                # Assicurati che i valori siano in range [0, 1]
                x_final = max(0, min(1, x_final))
                y_final = max(0, min(1, y_final))
                
                rotated_corners.append(x_final)
                rotated_corners.append(y_final)
            
            # Formato OBB YOLO: classe x1 y1 x2 y2 x3 y3 x4 y4
            yolo_bbox = [0] + rotated_corners
            
            # Per il controllo delle sovrapposizioni, usiamo ancora il rettangolo non ruotato
            pixel_bbox = [x1, y1, x2, y2]
            
            return result_img, yolo_bbox, pixel_bbox
    
    # Se dopo tutti i tentativi non è stato possibile evitare sovrapposizioni
    # Ritorna l'ultima posizione generata
    result_img = bg_img.copy()
    
    if card_img.mode == 'RGBA':
        mask = card_img.split()[3]
    else:
        mask = None
        
    result_img.paste(card_img, (paste_x, paste_y), mask)
    
    # Assicurati che la bounding box sia all'interno dell'immagine
    x1 = max(0, paste_x)
    y1 = max(0, paste_y)
    x2 = min(bg_width, paste_x + rotated_width)
    y2 = min(bg_height, paste_y + rotated_height)
    
    # Calcola il centro della carta
    center_x = paste_x + rotated_width / 2
    center_y = paste_y + rotated_height / 2
    
    # Calcola i quattro angoli prima della rotazione (relativo al centro)
    half_width = original_width / 2
    half_height = original_height / 2
    corners = [
        [-half_width, -half_height],  # Top-left
        [half_width, -half_height],   # Top-right
        [half_width, half_height],    # Bottom-right
        [-half_width, half_height]    # Bottom-left
    ]
    
    # Converti l'angolo in radianti (per il calcolo della rotazione)
    angle_rad = math.radians(-angle_deg)  # Negativo perché PIL ruota in senso antiorario
    
    # Applica la rotazione e trasla al centro
    rotated_corners = []
    for x, y in corners:
        # Applica la rotazione
        x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        
        # Trasla al centro e normalizza per YOLO
        x_final = (center_x + x_rot) / bg_width
        y_final = (center_y + y_rot) / bg_height
        
        # Assicurati che i valori siano in range [0, 1]
        x_final = max(0, min(1, x_final))
        y_final = max(0, min(1, y_final))
        
        rotated_corners.append(x_final)
        rotated_corners.append(y_final)
    
    # Formato OBB YOLO: classe x1 y1 x2 y2 x3 y3 x4 y4
    yolo_bbox = [0] + rotated_corners
    
    pixel_bbox = [x1, y1, x2, y2]
    
    return result_img, yolo_bbox, pixel_bbox

def generate_dataset_image(cards_files, backgrounds_files, idx, save_dir, labels_dir, cards_per_image=None):
    """
    Genera un'immagine di dataset con più carte Pokemon su uno sfondo.
    Salva l'immagine e il file di label corrispondente.
    """
    if cards_per_image is None:
        if random.random() < 0.5:
            grid_rows = random.randint(5, 15)
            grid_cols = random.randint(10, 20)
            cards_per_image = grid_rows * grid_cols
            use_grid = True
        else:
            # For random placement, sometimes generate very few cards
            cards_per_image = random.choices(
                [1, 2, 3, random.randint(8, 15)],
                weights=[0.1, 0.1, 0.1, 0.7]  # 30% chance for 1-3 cards, 70% for many cards
            )[0]
            use_grid = False
    else:
        use_grid = False
    
    # Seleziona uno sfondo casuale
    bg_file = random.choice(backgrounds_files)
    bg_img = Image.open(bg_file).convert("RGBA")
    
    # Ridimensiona lo sfondo a una dimensione standard per YOLO
    target_size = (640, 640)  # dimensione standard per dataset YOLO
    bg_img = bg_img.resize(target_size, Image.LANCZOS)
    
    # Lista per tenere traccia delle bounding box esistenti
    existing_boxes = []
    
    # Lista per memorizzare le coordinate YOLO
    yolo_bboxes = []
    
    if use_grid:
        # Genera una griglia regolare di carte
        cell_width = bg_img.width // grid_cols
        cell_height = bg_img.height // grid_rows
        
        # Calcola la dimensione delle carte (leggermente più piccola della cella)
        card_scale = 0.99  # Lascia un piccolo spazio tra le carte
        card_width = int(cell_width * card_scale)
        card_height = int(cell_height * card_scale)
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Seleziona una carta casuale
                card_file = random.choice(cards_files)
                card_img = Image.open(card_file).convert("RGBA")
                
                # Ridimensiona la carta per adattarla alla cella
                original_width, original_height = card_img.size
                aspect_ratio = original_height / original_width
                new_card_width = card_width
                new_card_height = int(new_card_width * aspect_ratio)
                
                # Se l'altezza è troppo grande, ridimensiona in base all'altezza
                if new_card_height > card_height:
                    new_card_height = card_height
                    new_card_width = int(new_card_height / aspect_ratio)
                
                card_img = card_img.resize((new_card_width, new_card_height), Image.LANCZOS)
                
                # Calcola la posizione centrale nella cella
                paste_x = col * cell_width + (cell_width - new_card_width) // 2
                paste_y = row * cell_height + (cell_height - new_card_height) // 2
                
                # Incolla la carta sullo sfondo
                if card_img.mode == 'RGBA':
                    mask = card_img.split()[3]
                else:
                    mask = None
                
                bg_img.paste(card_img, (paste_x, paste_y), mask)
                
                # Calcola la bounding box in formato YOLO (normalizzata)
                x1 = paste_x / bg_img.width
                y1 = paste_y / bg_img.height
                x2 = (paste_x + new_card_width) / bg_img.width
                y2 = y1
                x3 = x2
                y3 = (paste_y + new_card_height) / bg_img.height
                x4 = x1
                y4 = y3
                
                # Formato OBB YOLO: classe x1 y1 x2 y2 x3 y3 x4 y4
                yolo_bbox = [0, x1, y1, x2, y2, x3, y3, x4, y4]
                yolo_bboxes.append(yolo_bbox)
                
                # Aggiungi la bounding box all'elenco delle esistenti (per compatibilità)
                pixel_bbox = [paste_x, paste_y, paste_x + new_card_width, paste_y + new_card_height]
                existing_boxes.append(pixel_bbox)
    else:
        # Usa il posizionamento casuale originale
        for _ in range(cards_per_image):
            # Seleziona una carta casuale
            card_file = random.choice(cards_files)
            card_img = Image.open(card_file).convert("RGBA")

            # Determina il fattore di scala in base al numero di carte
            if cards_per_image == 1:
                scale_factor = random.uniform(0.7, 0.8)  # Carte molto grandi (70-80% dell'immagine)
            elif cards_per_image == 2:
                scale_factor = random.uniform(0.35, 0.45)  # Carte grandi (35-45% dell'immagine)
            elif cards_per_image == 3:
                scale_factor = random.uniform(0.25, 0.35)  # Carte medie (25-35% dell'immagine)
            else:
                scale_factor = random.uniform(0.08, 0.15)  # Carte piccole (8-15% dell'immagine)
            
            # Posiziona la carta sullo sfondo con il fattore di scala specificato
            result, yolo_bbox, bbox = place_card_on_background(
                card_img, bg_img, existing_boxes, scale_factor=scale_factor
            )
            bg_img = result
            
            # Aggiungi la bounding box all'elenco delle esistenti
            existing_boxes.append(bbox)
            
            # Memorizza le coordinate YOLO
            yolo_bboxes.append(yolo_bbox)
    
    # Salva l'immagine
    img_filename = f"image_{idx:06d}.jpg"
    img_path = os.path.join(save_dir, img_filename)
    bg_img = bg_img.convert("RGB")  # Converti in RGB per salvare come JPG
    bg_img.save(img_path, quality=95)
    
    # Salva il file di label YOLO OBB
    label_filename = f"image_{idx:06d}.txt"
    label_path = os.path.join(labels_dir, label_filename)
    
    with open(label_path, "w") as f:
        for bbox in yolo_bboxes:
            # Formato YOLO OBB: classe x1 y1 x2 y2 x3 y3 x4 y4
            formatted_bbox = [
                int(bbox[0]),  # classe (intero)
            ]
            # Aggiungi le 8 coordinate (4 punti) arrotondate a 6 decimali
            for i in range(1, 9):
                formatted_bbox.append(round(float(bbox[i]), 6))
                
            line = " ".join(map(str, formatted_bbox))
            f.write(line + "\n")
            
    return img_path, label_path

def create_yaml_file():
    """Crea il file YAML per l'addestramento di YOLOv8."""
    data = {
        'train': './train/images',  # Percorso relativo per Ultralytics HUB
        'val': './val/images',
        'test': './test/images',
        'nc': 1,  # Numero di classi
        'names': ['pokemon_card']  # Nome della classe
    }
    
    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print("✓ File YAML creato con successo")

def split_dataset(total_images=10000, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Calcola la suddivisione del dataset in train, validation e test.
    Restituisce il numero di immagini per ogni split.
    """
    train_images = int(total_images * train_ratio)
    val_images = int(total_images * val_ratio)
    test_images = total_images - train_images - val_images
    
    return {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

def create_zip_file():
    """Crea un file ZIP del dataset."""
    zip_filename = 'pokemon_cards_dataset.zip'
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Crea una struttura come richiesta da Ultralytics HUB
        for split in ['train', 'val', 'test']:
            # Aggiungi immagini
            imgs_dir = f'dataset/images/{split}'
            for img_file in os.listdir(imgs_dir):
                img_path = os.path.join(imgs_dir, img_file)
                # Percorso nel file ZIP: train/images/file.jpg
                zipf.write(img_path, f'{split}/images/{img_file}')
            
            # Aggiungi labels
            labels_dir = f'dataset/labels/{split}'
            for label_file in os.listdir(labels_dir):
                label_path = os.path.join(labels_dir, label_file)
                # Percorso nel file ZIP: train/labels/file.txt
                zipf.write(label_path, f'{split}/labels/{label_file}')
        
        # Aggiungi il file YAML
        zipf.write('dataset/data.yaml', 'data.yaml')
    
    print(f"✓ File ZIP '{zip_filename}' creato con successo")
    return zip_filename

def verify_labels(labels_dir):
    """Verifica che i file di label siano formattati correttamente per OBB."""
    invalid_files = []
    
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 9:  # Formato OBB: classe + 8 coordinate
                    invalid_files.append(label_file)
                    break
                try:
                    # Controlla che i valori siano numeri e nel range corretto
                    class_id = int(parts[0])
                    
                    # Verifica che tutte le coordinate siano nel range [0,1]
                    for i in range(1, 9):
                        coord = float(parts[i])
                        if coord < 0 or coord > 1:
                            invalid_files.append(label_file)
                            break
                            
                except ValueError:
                    invalid_files.append(label_file)
                    break
    
    if invalid_files:
        print(f"ATTENZIONE: Trovati {len(invalid_files)} file di label non validi")
        return False
    else:
        print("✓ Tutti i file di label sono formattati correttamente per OBB")
        return True

def main():
    """Funzione principale per generare il dataset."""
    print("Generazione dataset YOLO per rilevamento carte Pokemon")
    
    # Definisci percorsi
    cards_dir = 'carte_pokemon'
    backgrounds_dir = 'background'
    
    # Verifica esistenza cartelle
    if not os.path.exists(cards_dir):
        raise FileNotFoundError(f"La cartella {cards_dir} non esiste")
    if not os.path.exists(backgrounds_dir):
        raise FileNotFoundError(f"La cartella {backgrounds_dir} non esiste")
    
    # Crea struttura directory
    create_directory_structure()
    
    # Ottieni file
    cards_files = get_card_files(cards_dir)
    backgrounds_files = get_background_files(backgrounds_dir)
    
    if not cards_files:
        raise ValueError(f"Nessuna immagine trovata nella cartella {cards_dir}")
    if not backgrounds_files:
        raise ValueError(f"Nessuna immagine trovata nella cartella {backgrounds_dir}")
    
    # Calcola la suddivisione del dataset
    total_images = 10000  # Numero totale di immagini richiesto
    split_counts = split_dataset(total_images)
    
    print(f"Generazione di {total_images} immagini:")
    print(f" - Train: {split_counts['train']} immagini")
    print(f" - Validation: {split_counts['val']} immagini")
    print(f" - Test: {split_counts['test']} immagini")
    
    # Genera il dataset
    current_idx = 0
    
    # Genera immagini di train
    print("\nGenerazione immagini di train...")
    for i in tqdm(range(split_counts['train'])):
        generate_dataset_image(
            cards_files, 
            backgrounds_files, 
            current_idx,
            'dataset/images/train',
            'dataset/labels/train'
        )
        current_idx += 1
    
    # Genera immagini di validation
    print("\nGenerazione immagini di validation...")
    for i in tqdm(range(split_counts['val'])):
        generate_dataset_image(
            cards_files, 
            backgrounds_files, 
            current_idx,
            'dataset/images/val',
            'dataset/labels/val'
        )
        current_idx += 1
    
    # Genera immagini di test
    print("\nGenerazione immagini di test...")
    for i in tqdm(range(split_counts['test'])):
        generate_dataset_image(
            cards_files, 
            backgrounds_files, 
            current_idx,
            'dataset/images/test',
            'dataset/labels/test'
        )
        current_idx += 1
    
    # Verifica i file di label
    print("\nVerifica dei file di label...")
    verify_labels('dataset/labels/train')
    
    # Crea file YAML
    create_yaml_file()
    
    # Crea file ZIP
    zip_filename = create_zip_file()
    
    print("\nGenerazione dataset completata!")
    print(f"Il dataset è stato salvato nella cartella 'dataset' e compresso in '{zip_filename}'")
    print("\nRiepilogo:")
    print("- Modello consigliato: YOLOv8n (Nano) per esecuzione ottimizzata su browser")
    print("- Per l'addestramento: yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640")
    print("- Per la conversione in formato web: yolo export model=runs/train/best.pt format=tfjs")
    
    return True

if __name__ == "__main__":
    main()