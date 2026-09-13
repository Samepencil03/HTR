# End-to-End Handwritten Text Recognition (HTR) Pipeline

An end-to-end modular handwritten text recognition pipeline built with Python. It accepts handwritten document images, detects line regions, transcribes handwriting with TrOCR, cleans and understands the text using a local Gemma model via Ollama, and converts the final text into spoken speech with TTS.

---

## 🏗️ Architecture

```
IMAGE FILE
    │
    ▼
[1/5] PaddleOCR Text Detection (detector.py)
    │  - Detects bounding box regions
    │  - Sorts boxes top-to-bottom, left-to-right (reading order)
    │  - Saves text line crops to crops/ (in --debug mode)
    ▼
[2/5] Handwritten Text Recognition (htr.py)
    │  - TrOCR Model (microsoft/trocr-base-handwritten)
    │  - Transcribes each line crop in reading order
    ▼
[3/5] Raw Text Assembly (main.py)
    │  - Joins line transcriptions preserving line breaks (\n)
    ▼
[4/5] Local Gemma LLM Processing (llm.py)
    │  ├─ Stage A: Correction (Fixes OCR typos & punctuation without changing meaning)
    │  └─ Stage B: Understanding (Cleans and formats output without hallucination)
    ▼
[5/5] Text-To-Speech & Output (tts.py & output/result.txt)
    │  - Speaks final text via pyttsx3 / espeak-ng / spd-say
    │  - Saves output to output/result.txt
```

---

## 📋 Prerequisites

Before running the pipeline, ensure your system has:

1. **Python 3.9 – 3.12**
2. **Ollama** installed on your system ([https://ollama.com](https://ollama.com))
3. *(Optional for Linux Audio)* System speech synthesis engine (`espeak-ng` or `espeak`):
   ```bash
   sudo apt-get update && sudo apt-get install -y espeak-ng
   ```

---

## 🚀 Step-by-Step Installation

### 1. Clone or Open the Repository
```bash
cd paddleocr-test
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Ollama and Pull a Gemma Model
Ensure the Ollama server is running and download a local Gemma model:
```bash
ollama serve &
ollama pull gemma2:2b
```

---

## 💡 How to Run the Program

### Basic Command
Run the pipeline on any image containing handwritten text:
```bash
python main.py /path/to/handwritten_image.png
```

### Debug Mode (`--debug`)
Save line crops to `crops/`, generate bounding box overlay image to `output/detected_boxes.png`, and view detailed step-by-step logs:
```bash
python main.py input/sample_handwritten.png --debug
```

### Custom Model (`--model`)
Specify a specific local Gemma or Ollama model name:
```bash
python main.py input/sample_handwritten.png --model gemma2:2b
```
Or set the environment variable:
```bash
export GEMMA_MODEL=gemma2:2b
python main.py input/sample_handwritten.png
```

---

## 📁 Output Files

When execution completes, outputs are saved in the `output/` directory:

- **Final Processed Text**: `output/result.txt`
- **Detected Bounding Boxes Visualization (Debug Mode)**: `output/detected_boxes.png`
- **Individual Line Crops (Debug Mode)**: `crops/crop_000.png`, `crops/crop_001.png`, ...

---

## 📂 Project Structure

```
.
├── main.py                # Main pipeline orchestrator (stages 1 to 5)
├── detector.py            # PaddleOCR text detection & reading-order sorting
├── htr.py                 # Dedicated TrOCR handwriting recognition model
├── llm.py                 # Ollama connection & 2-stage LLM correction/understanding
├── tts.py                 # Offline Linux text-to-speech engine
├── create_sample_image.py # Helper script to generate a test image
├── requirements.txt       # Python dependencies list
├── README.md              # Project documentation
├── input/                 # Directory for input images
├── output/                # Directory for output result.txt & visualizations
├── crops/                 # Directory for temporary text line crops (debug mode)
├── web_ui.py              # FastAPI web interface
└── tmp_uploads/           # Temporary upload folder (runtime)
```

---

## 🌐 Web Interface (FastAPI)

A lightweight web UI is provided to run the pipeline without using the terminal.

### Additional Dependencies
The web UI requires a few extra packages. Install them with:
```bash
pip install fastapi uvicorn python-multipart
```

### Running the Server
Start the FastAPI server using **uvicorn**:
```bash
uvicorn web_ui:app --host 0.0.0.0 --port 8000
```
The server will be accessible at `http://localhost:8000`. Open this URL in a browser, upload an image, and the page will display the raw and final text results.

### Endpoints
- `GET /` – Serves the HTML upload page.
- `POST /process` – Accepts an image file, runs the full pipeline, and returns JSON with `raw_text` and `final_text`.

---

## 🔧 Troubleshooting

- **Ollama server connection error**:
  Ensure Ollama is running by executing `ollama serve` in a terminal window.
- **Model memory limit error**:
  If a model is too large for system RAM, pull a smaller model such as `ollama pull gemma2:2b` or `ollama pull qwen2.5:0.5b` and pass `--model gemma2:2b`.
- **No text-to-speech audio**:
  If running on a headless server without audio hardware, the pipeline will log a notice and still output the full text result cleanly to `output/result.txt`.
