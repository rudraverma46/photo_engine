📸 Local Semantic Photo Search Engine

A privacy-first, completely local alternative to cloud-based photo storage. This multi-model AI pipeline processes images to generate rich, contextual metadata, allowing for intelligent, natural language image retrieval without sending your private data to external servers.

🧠 Architecture

This project utilizes three distinct AI models working in tandem to understand and index images:

Facial Recognition (facenet-pytorch): Uses MTCNN for face detection and InceptionResnetV1 to generate facial embeddings. It interactively learns and remembers human faces.

Environmental Context (LLaVA via Ollama): A Vision-Language Model that analyzes the background, lighting, and objects in the image, strictly ignoring the people to provide scene context.

Semantic Search (sentence-transformers + FAISS): Converts the synthesized text metadata (people + environment) and user search queries into dense mathematical vectors, using Cosine Similarity for lightning-fast, highly accurate retrieval.

⚙️ Prerequisites

Python 3.8+

Ollama installed locally

Git

🛠️ Installation

Clone the repository:

git clone [https://github.com/rudraverma46/photo_engine.git](https://github.com/rudraverma46/photo_engine.git)
cd photo_engine


Install the required Python packages:

pip install -r requirements.txt


Pull the required vision model via Ollama:

ollama pull llava


🚀 Usage

Phase 1: Indexing & Learning

Create a directory named my_photos in the root of the project and place your images inside. Run the smart indexer to process the images.

If the script detects a face it hasn't seen before, it will temporarily pause, display the cropped face using your system's default image viewer, and ask you to name the person in the terminal.

python3 smart_indexer.py


Outputs generated: known_faces.json (facial embeddings database) and image_metadata.json (environmental and subject context).

Phase 2: Searching

Once the metadata is built, run the search engine. The script will compile the text descriptions into a FAISS vector database for instant retrieval.

You can search by person, environment, or a combination of both.

Example: "John standing on a beach"

Example: "A red sports car on the road"

python3 search.py
