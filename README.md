# Library Recommendation System

This project is an AI-powered book recommendation system featuring a complete pipeline, including embedding generation, autoencoder training, clustering, FAISS vector database, PostgreSQL storage, ClearML experiment tracking, and a Streamlit frontend.

## System Architecture
The library recommendation system consists of two main workflow: **Embedding Construction** and **Recommendation & Interaction**.


## Usage 
### 1. Clone the Repository

```bash
git clone https://github.com/jennycs0830/library_recommendation_system.git
cd library_recommendation_system
```

### 2. Install Python Dependencies
It is recommended to use a virtual environment. Then install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set Up ClearML

1. Register at [ClearML](https://app.clear.ml/login).
2. Navigate to: `Settings > Workspace > Create new credentials`
3. Copy the generated configuration block.
4. Run the setup:
```bash
clearml-init
```
5. Paste the block when prompted. This will generate the `.clearml.conf` config file.

### 4. Upload Data to ClearML
```bash
# books csv file
python clearml_dataset_upload.py --dataset_name books_with_intro --upload_files data/books_with_intro.csv
```
### 5. Run preprocessing
Run data cleaning, embedding generation using SentenceTransformer, and train/test split:
```bash
cd train
python preprocessing.py
```
**Uploaded ClearML datasets ([dataset_project]-[dataset_name]-[file_name]):**
- `AI_recommender-books_with_intro-books_with_intro.csv`
- `AI_recommender-books_with_intro_cleaned-books_with_intro_cleaned.csv`
- `AI_recommender-embeddings_db-all_embeddings.pt`
- `AI_recommender-embeddings_db-train_embeddings.pt`
- `AI_recommender-embeddings_db-test_embeddings.pt`

### 6. Train AutoEncoder
```bash
python train_AutoEncoder.py
```
- Output model file will be stored in the `train/` directory.

### 7. Train Clustering Model & Generate Encoded Embeddings
Check the autoencoder model filename locally, then run:
```bash
python train_DC.py --autoencoder_file [autoencoder_model_filename]
```
**Uploaded ClearML dataset:**
- `AI_recommender-encoded_embedding_db-encoded_embeddings.csv`

### 8. Install and Set Up PostgreSQL
Create a PostgreSQL User and Database
```bash
sudo -u postgres psql
```

Then in the PostgreSQL shell:
```sql
CREATE USER library_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE library_db;
GRANT ALL PRIVILEGES ON DATABASE library_db TO library_user;
\q
```

### 9. Create Databases (PostgreSQL + FAISS)
Initial Postgres
```bash
cd ../database
python create_db.py
python create_faiss_db.py
```

### 10. Build Docker Images
Ensure Docker is installed and running.

#### Autoencoder API Image
```bash
docker build -f docker/Dockerfile.autoencoder -t autoencoder-api .
```
#### Clustering API Image
```bash
docker build -f docker/Dockerfile.clustering -t clustering-api .
```

### 11. Run Docker Containers

#### Autoencoder API (http://localhost:8002/encode)
```bash
docker run -d --name autoencoder-api -p 8002:8002 autoencoder-api
```

#### Clustering API (http://localhost:8003/encode)
```bash
docker run -d --name clustering-api -p 8003:8003 clustering-api
```
### 12. Configure API Keys and DB Credentials

Create a `.env` file in the `components` folder. Example:
```bash
DB_HOST=[your host]
DB_PORT=[your port]
DB_USER=[your username]
DB_PASSWORD=[your password]
OPENAI_API_KEY=[your api key]
```

### 12. Launch Streamlit App
```bash
streamlit run app.py
```
