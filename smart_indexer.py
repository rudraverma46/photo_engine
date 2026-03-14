import os
import json
import base64
import requests
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# --- Configuration ---
IMAGE_DIR = './my_photos'
KNOWN_FACES_FILE = 'known_faces.json'
METADATA_FILE = 'image_metadata.json'
OLLAMA_API_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"
SIMILARITY_THRESHOLD = 0.75 # Higher = stricter face matching

print("[*] Booting up neural networks. This will take a few seconds...")
# Initialize PyTorch device (use CUDA if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# MTCNN detects where the face is in the picture
mtcnn = MTCNN(keep_all=True, device=device)
# InceptionResnetV1 maps the face into a 512-dimensional vector
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_cosine_similarity(vec1, vec2):
    """Calculates how closely two face vectors match."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def encode_image_for_ollama(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_environmental_context(image_path):
    """Asks local LLaVA to describe the background, ignoring people."""
    print("    -> Asking LLaVA to analyze the environment...")
    base64_img = encode_image_for_ollama(image_path)
    
    prompt = """
    Describe the environment, setting, objects, weather, and lighting in this image in high detail. 
    Do NOT mention the people. Focus entirely on where this is and what objects are present.
    """
    
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [base64_img],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"    [!] Failed to reach Ollama: {e}")
        return "Unknown environment."

def process_faces(image, known_faces):
    """Detects faces, compares them to known faces, and asks you if unknown."""
    img_cropped_list = mtcnn(image)
    names_in_photo = []
    
    if img_cropped_list is not None:
        # Get embeddings for all detected faces in the image
        embeddings = resnet(img_cropped_list.to(device)).detach().cpu().numpy()
        
        # We also need the bounding boxes so we can crop the face to show you
        boxes, _ = mtcnn.detect(image)
        
        for i, face_embedding in enumerate(embeddings):
            best_match_name = None
            highest_sim = -1
            
            # Compare this face against every face we already know
            for known_name, known_emb in known_faces.items():
                sim = get_cosine_similarity(face_embedding, known_emb)
                if sim > highest_sim:
                    highest_sim = sim
                    best_match_name = known_name
            
            if highest_sim > SIMILARITY_THRESHOLD:
                print(f"    -> Recognized: {best_match_name} (Confidence: {highest_sim:.2f})")
                names_in_photo.append(best_match_name)
            else:
                # It's a new person! Let's ask you who it is.
                box = boxes[i]
                face_crop = image.crop((box[0], box[1], box[2], box[3]))
                
                print("    [!] Unknown face detected! Popping up image viewer...")
                face_crop.show() # Opens Linux Mint's default image viewer
                
                name = input("    Who is this? (Type name, or press Enter to skip/ignore): ").strip()
                if name:
                    known_faces[name] = face_embedding.tolist() # Save for next time
                    names_in_photo.append(name)
                    print(f"    -> Saved {name} to memory.")
                else:
                    names_in_photo.append("Unknown Person")
    
    return names_in_photo, known_faces

def main():
    if not os.path.exists(IMAGE_DIR):
        print(f"[!] Please create {IMAGE_DIR} and add some photos containing people.")
        return

    # Load Databases
    known_faces = {}
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, 'r') as f:
            known_faces = json.load(f)
            
    metadata_db = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata_db = json.load(f)

    # Process each image
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            filepath = os.path.join(IMAGE_DIR, filename)
            
            if filepath in metadata_db:
                continue # Skip images we've already processed
                
            print(f"\n[*] Processing: {filename}")
            try:
                img = Image.open(filepath).convert('RGB')
            except Exception as e:
                print(f"    [!] Error opening image: {e}")
                continue

            # 1. Handle Faces
            names, known_faces = process_faces(img, known_faces)
            
            # 2. Handle Environment
            environment_desc = get_environmental_context(filepath)
            
            # 3. Combine and Save
            people_str = ", ".join(names) if names else "No people detected"
            final_metadata = f"People present: {people_str}. Context: {environment_desc}"
            
            metadata_db[filepath] = final_metadata
            print(f"    -> Final Tag: {final_metadata}")
            
            # Save progress after every image
            with open(KNOWN_FACES_FILE, 'w') as f:
                json.dump(known_faces, f, indent=4)
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata_db, f, indent=4)

    print("\n[+] All images processed successfully! Metadata is ready for search.")

if __name__ == '__main__':
    main()
