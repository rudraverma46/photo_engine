import os
import json
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# Constants
IMAGE_DIR = './my_photos'
EMBEDDINGS_DB_PATH = 'embeddings.npy'
INDEX_PATH = 'faiss_index.index'
PATHS_FILE = 'image_paths.json'

# Load pre-trained CLIP model
print("[*] Loading CLIP Vision-Language Model...")
model = SentenceTransformer('clip-ViT-B-32')

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    # FIX 1: The correct method is encode(), not encode_image()
    image_tensor = model.encode(image) 
    return image_tensor

# Function to build or load the embeddings database
def build_embeddings_db(image_dir, db_path, index_path, paths_file):
    # FIX 2: Check if the paths file exists before trying to load
    if os.path.exists(db_path) and os.path.exists(index_path) and os.path.exists(paths_file):
        print("[*] Loading existing FAISS database...")
        feature_list = np.load(db_path)
        # FIX 3: faiss.read_index takes a string, not a file object
        index = faiss.read_index(index_path) 
        with open(paths_file, 'r') as f:
            image_paths = json.load(f)
    else:
        print("[*] Building new embeddings database. This may take a moment...")
        feature_list = []
        image_paths = []

        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_path = os.path.join(root, file)
                    features = extract_features(image_path)
                    feature_list.append(features)
                    image_paths.append(image_path)

        if not feature_list:
            print("[!] No images found to process.")
            return None, None, None

        feature_list = np.array(feature_list)
        faiss.normalize_L2(feature_list)

        index = faiss.IndexFlatIP(feature_list.shape[1])
        index.add(feature_list)

        np.save(db_path, feature_list)
        faiss.write_index(index, index_path)
        
        # FIX 4: Save the image paths so we know which file is which next time
        with open(paths_file, 'w') as f:
            json.dump(image_paths, f)

    return index, feature_list, image_paths

# Function to search for images based on a query
def search_images(query, index, image_paths):
    # FIX 5: Encode the text directly! Do not pass text into Image.open()
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    D, I = index.search(query_embedding, k=3)  # Find top 3 matches

    results = []
    for i in range(len(I[0])):
        if I[0][i] != -1: # Ensure it's a valid match
            results.append((image_paths[I[0][i]], D[0][i]))

    return results

# Main function
def main():
    if not os.path.exists(IMAGE_DIR):
        print(f"[!] Directory '{IMAGE_DIR}' not found. Please create it and add photos.")
        return

    index, feature_list, image_paths = build_embeddings_db(IMAGE_DIR, EMBEDDINGS_DB_PATH, INDEX_PATH, PATHS_FILE)
    
    if index is None:
        return

    print("\n--- Local Semantic Search Active ---")
    while True:
        query = input("\nEnter search context (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        results = search_images(query, index, image_paths)
        print("\n[ Top Matches ]")
        for result in results:
            print(f"File: {result[0]} | Similarity: {result[1]:.4f}")

if __name__ == '__main__':
    main()
