
#  SmartStyle: Virtual Try-On using YOLOv8 and OpenAI Embeddings

SmartStyle is a real-time AI-powered virtual try-on application built by **siddhanthnagrath1**, using YOLOv8 for object detection and OpenAI embeddings for intelligent similarity matching. It allows users to detect fashion items and try them on virtually using a simple web interface.

---

##  Setup Instructions

###  Clone the Repository
```bash
git clone https://github.com/siddhanthnagrath1/SmartStyle.git
cd SmartStyle
```

###  Set up the Python3 environment
```bash
python3 -m venv myntra_env
source ./myntra_env/bin/activate
```

###  Install Required Packages
```bash
pip install -r ./requirement.txt
```

---

##  System Dependencies

### For macOS:
```bash
brew install ffmpeg
```

### For Ubuntu/Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

---

##  Get Additional Files

### 1. Embedding Images Directory
```bash
cd app
unzip embedding_images.zip -d embedding_images
```

### 2. YOLOv8 Trained Model
- Download `best.pt.zip` from:  
  [https://github.com/siddhanthnagrath1/SmartStyle-YOLOv8](https://github.com/siddhanthnagrath1/SmartStyle-YOLOv8)
- Unzip to extract `best.pt`
- Place the file in the root of the `SmartStyle` project directory

---

##  Run the FastAPI Server
```bash
uvicorn server:app --reload --port 8000
```

---

##  Launch the Frontend
- Navigate to the project directory
- Right-click on `index.html` and open it in any modern browser

---

##  Configuration

1. Create your OpenAI API token.
2. Open the file `./config.yaml`.
3. Replace the value of `openai_api_key` with your actual key:
```yaml
openai_api_key: "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---


