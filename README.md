# End-to-End Handwritten Text Recognition (HTR) Pipeline & Web Dashboard

An end-to-end, modular handwritten text recognition (HTR) pipeline and interactive web dashboard. It processes handwritten document images, detects and merges text line regions using PaddleOCR, transcribes handwriting in batch using Hugging Face TrOCR, corrects and cleans the transcriptions using a local Gemma LLM via Ollama, and synthesizes spoken speech output (`output/speech.wav`).

---

## 🏗️ Architecture & Pipeline Flow

```
HANDWRITTEN IMAGE
       │
       ▼
[1/5] PaddleOCR Text Detection (detector.py)
       │  - Detects text bounding boxes
       │  - Clusters adjacent word boxes into continuous text lines
       │  - Applies dynamic padding (8% edge margin) to prevent letter clipping
       │  - Sorts line crops in natural reading order (top-to-bottom, left-to-right)
       ▼
[2/5] Batch Handwritten Text Recognition (htr.py)
       │  - TrOCR Model (microsoft/trocr-base-handwritten)
       │  - Processes cropped line images in a single parallel batch pass
       ▼
[3/5] Raw Text Assembly (main.py)
       │  - Joins line transcriptions preserving line breaks (\n)
       ▼
[4/5] Local Gemma LLM Processing (llm.py)
       │  ├─ Stage A: Correction (Fixes OCR typos & punctuation without changing meaning)
       │  └─ Stage B: Understanding (Cleans and formats output into final representation)
       ▼
[5/5] Text-To-Speech & Export (tts.py & output/)
       │  - Synthesizes speech via pyttsx3 (SAPI5 on Windows, espeak-ng on Linux/macOS)
       │  - Saves transcript to output/result.txt
       │  - Exports audio speech file to output/speech.wav
```

---

## 🌟 Key Features

- **Cross-Platform**: Fully compatible with **Windows**, **Linux**, and **macOS**.
- **Text Line Merging**: Automatically combines word bounding boxes into full text lines for optimal TrOCR recognition accuracy.
- **Batch Recognition**: Processes multi-line document crops in parallel using Hugging Face TrOCR.
- **Local LLM Post-Processing**: Automatically connects to Ollama, auto-detects local Gemma models (`gemma2:2b`, `gemma3:4b`), and performs text correction and understanding.
- **WAV Speech Export**: Generates an audio speech file (`output/speech.wav`) alongside live browser audio playback.
- **Interactive Web Dashboard (FastAPI)**: Features real-time pipeline metrics, live speech player, visual bounding box overlays, line crops gallery, and an interactive drawing whiteboard.

---

## 📋 Prerequisites & System Requirements

- **Python**: 3.9 – 3.12 installed on your system.
- **Ollama**: Installed locally ([https://ollama.com](https://ollama.com)) with a Gemma model pulled (`ollama pull gemma2:2b`).
- **System Audio Engine (TTS)**:
  - **Windows**: Built-in Windows SAPI5 (no additional installation required).
  - **Linux**: Requires `espeak-ng` or `espeak`:
    ```bash
    sudo apt-get update && sudo apt-get install -y espeak-ng
    ```
  - **macOS**: Built-in speech engine / `espeak`.

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Samepencil03/HTR.git
cd HTR
```

### 2. Create and Activate Virtual Environment

#### 🐧 Linux & 🍎 macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 🪟 Windows (Command Prompt `cmd.exe`):
```cmd
python -m venv venv
venv\Scripts\activate
```

#### 🪟 Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🤖 Ollama & LLM Setup

Install Ollama for your operating system:
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows**: Download installer from [ollama.com/download/windows](https://ollama.com/download/windows)
- **macOS**: Download installer from [ollama.com/download/mac](https://ollama.com/download/mac)

Start the Ollama server and pull the Gemma model:
```bash
# Start server (if not running as a background service)
ollama serve

# Pull local Gemma model (in another terminal)
ollama pull gemma2:2b
```

---

## 💻 Running via Command Line (CLI)

Run the end-to-end pipeline on any handwritten image file:

#### Linux / macOS:
```bash
python main.py input/sample_handwritten.png
```

#### Windows:
```cmd
python main.py input\sample_handwritten.png
```

### Command Flags:
- **Debug Mode (`--debug`)**: Saves cropped line images to `crops/`, draws bounding box overlays to `output/detected_boxes.png`, and logs step-by-step debug information.
  ```bash
  python main.py input/sample_handwritten.png --debug
  ```
- **Custom Model (`--model`)**: Specify a custom local Ollama model.
  ```bash
  python main.py input/sample_handwritten.png --model gemma2:2b
  ```

---

## 🌐 Interactive Web Dashboard (FastAPI)

Launch the FastAPI web server:

```bash
python -m uvicorn web_ui:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

### Dashboard Features:
1. **Interactive Canvas / Whiteboard**: Draw handwritten text directly on the canvas or upload images (`.png`, `.jpg`, `.jpeg`, `.webp`).
2. **Real-Time Metrics**: View detected line counts, overall processing duration, active HTR model, and Gemma LLM model.
3. **Visual Overlays & Crop Gallery**: Inspect detected text line bounding box visualizations and view individual cropped line segments.
4. **Built-in Speech Player**: Listen to synthesized speech directly in your browser or download the `.wav` audio file.

---

## 🔗 REST API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | GET | Renders the full interactive web dashboard HTML interface. |
| `POST /process` | POST | Uploads an image file (`multipart/form-data`) and runs detection, TrOCR recognition, Gemma LLM cleanup, and TTS speech synthesis. Returns JSON with raw text, final text, metrics, and base64 images. |
| `GET /health` | GET | Returns server status and model warmup readiness (`{"status": "ok", "models_ready": true}`). |
| `GET /audio` | GET | Downloads or streams the generated WAV speech output (`output/speech.wav`). |

---

## 📁 Output Artifacts

All pipeline outputs are automatically saved to the following directories:

- **Text Output**: `output/result.txt` (Contains both raw TrOCR transcription and final Gemma LLM text)
- **Speech Audio**: `output/speech.wav` (Synthesized audio file)
- **Bounding Box Visualization**: `output/detected_boxes.png` (Visual overlay showing merged text lines)
- **Line Crops**: `crops/crop_000.png`, `crops/crop_001.png`, ... (Cropped text line images)

---

## 🛠️ Troubleshooting

- **PyTorch Memory / Install Issues**: Install CPU-only PyTorch if GPU memory or space is limited:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- **Ollama Connection Warning**: Ensure `ollama serve` is running on `http://localhost:11434`.
- **TTS Audio Errors on Linux**: Install `espeak-ng` (`sudo apt install espeak-ng`). On Windows, pyttsx3 uses native Windows SAPI5 automatically.
