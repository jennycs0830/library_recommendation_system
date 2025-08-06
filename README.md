
# 📚 Library Recommendation System

This project is an AI-powered book recommendation system with a full-stack pipeline including embedding generation, autoencoder training, clustering, FAISS vector DB, PostgreSQL storage, ClearML experiment tracking, and a Streamlit frontend.

---

## 🚀 Setup Instructions

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/jennycs0830/library_recommendation_system.git
cd library_recommendation_system
```

### 2. 📦 Install Python Dependencies

Make sure you're using a virtual environment, then run:

```bash
pip install -r requirements.txt
```

### 3. 🛠️ Build Docker Images

Ensure Docker is installed and running.

#### 🔹 Autoencoder API Image

```bash
docker build -f docker/Dockerfile.autoencoder -t autoencoder-api .
```

#### 🔹 Clustering API Image

```bash
docker build -f docker/Dockerfile.clustering -t clustering-api .
```

### 4. 🧱 Run Docker Containers

#### 🔹 Autoencoder API (http://localhost:8002/encode)

```bash
docker run -d --name autoencoder-api -p 8002:8002 autoencoder-api
```

#### 🔹 Clustering API (http://localhost:8003/encode)

```bash
docker run -d --name clustering-api -p 8003:8003 clustering-api
```

### 5. ☁️ Set Up ClearML

1. Go to [ClearML Login](https://app.clear.ml/login) and register.
2. Navigate to: `Settings > Workspace > Create new credentials`
3. Copy the generated configuration block.
4. Run the setup:

```bash
clearml-init
```

Paste the block when prompted. This will generate the `.clearml.conf` config file.

### 6. 🗃️ Install and Set Up PostgreSQL

#### 🔹 On Ubuntu / Debian

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

#### 🔹 Create a PostgreSQL User and Database

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

#### 🔹 Verify PostgreSQL Setup

```bash
psql -U library_user -d library_db -h localhost
```

When connected, type:

```sql
\dt
\q
```

### 7. 📚 Generate Initial Embeddings

#### 🔸 Upload book data to ClearML

```bash
python clearml_dataset_upload.py --dataset_name "books_with_intro" --upload_files "data/books_with_intro.csv"
```

#### 🔸 Run preprocessing (cleaning, SentenceTransformer, train/test split)

```bash
cd train
python preprocessing.py
```

**Uploaded ClearML datasets:**

- `AI_recommender-books_with_intro-books_with_intro.csv`
- `AI_recommender-books_with_intro_cleaned-books_with_intro_cleaned.csv`
- `AI_recommender-embeddings_db-all_embeddings.pt`
- `AI_recommender-embeddings_db-train_embeddings.pt`
- `AI_recommender-embeddings_db-test_embeddings.pt`

### 8. 🧠 Train AutoEncoder

```bash
python train_AutoEncoder.py
```

- Output model file will be stored in the `train/` directory.

### 9. 🧪 Train Clustering Model & Generate Encoded Embeddings

```bash
python train_DC.py --autoencoder_file [autoencoder_model_filename]
```

**Uploaded ClearML dataset:**

- `AI_recommender-encoded_embedding_db-encoded_embeddings.csv`

### 10. 🗂️ Create Databases (PostgreSQL + FAISS)

```bash
cd ../database
python create_db.py
python create_faiss_db.py
```

### 11. 🔧 Configure API Keys and DB Credentials

- **OpenAI API Key:** `components/register.py` (line 11)
- **PostgreSQL credentials:** `components/utils.py` (lines 46–47)

### 12. 🖥️ Launch Streamlit App

```bash
streamlit run app.py
```

Visit the running app at: [http://localhost:8501](http://localhost:8501)

---

✅ You're all set! The full recommendation system should now be functional with Docker, PostgreSQL, FAISS, ClearML tracking, and a Streamlit frontend.
