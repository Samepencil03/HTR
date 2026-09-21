import os
import shutil
import uuid
import threading
import glob
import time
import base64
from PIL import Image as PilImage
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import pipeline functions
from detector import detect_text
from htr import recognize_batch
from llm import process_text_pipeline, get_available_model
from tts import speak

app = FastAPI(title="Handwritten Text Recognition Web UI & Dashboard")

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

TEMP_DIR = "tmp_uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

_models_ready = False

@app.on_event("startup")
async def startup_event():
    global _models_ready
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("crops", exist_ok=True)

    for f in glob.glob(os.path.join(TEMP_DIR, "*")):
        try:
            os.remove(f)
        except Exception:
            pass

    def _warmup():
        global _models_ready
        try:
            print("[Startup] Warming up PaddleOCR detector...")
            from detector import get_detector
            get_detector()
            print("[Startup] Warming up TrOCR recogniser...")
            from htr import HTRRecognizer
            HTRRecognizer()._ensure_loaded()
            print("[Startup] All models ready.")
        except Exception as e:
            print(f"[Startup] Model warmup notice: {e}")
        finally:
            _models_ready = True

    threading.Thread(target=_warmup, daemon=True).start()

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "models_ready": _models_ready
    })

@app.get("/audio")
async def get_audio():
    audio_path = os.path.join("output", "speech.wav")
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        return FileResponse(audio_path, media_type="audio/wav", filename="speech.wav")
    raise HTTPException(status_code=404, detail="Audio file not generated yet")

HTML_PAGE = """
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Handwritten Text Recognition Dashboard</title>
    <style>
        :root {
            --bg-primary: #121420;
            --bg-card: rgba(255, 255, 255, 0.05);
            --bg-card-active: rgba(255, 255, 255, 0.09);
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --accent-light: rgba(99, 102, 241, 0.15);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --border: rgba(255, 255, 255, 0.1);
            --radius: 12px;
            --spacing: 1.25rem;
        }
        body {
            margin: 0; padding: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary); color: var(--text-primary);
            min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; box-sizing: border-box;
        }
        .container {
            width: 100%; max-width: 1040px; display: flex; flex-direction: column; gap: var(--spacing);
        }
        header { text-align: center; margin-bottom: 0.5rem; }
        header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #a5b4fc, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        header p { margin: 0.4rem 0 0; color: var(--text-secondary); font-size: 0.95rem; }
        
        #server-banner {
            display: none; padding: 0.75rem 1.25rem; border-radius: var(--radius);
            background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b;
            color: #fbbf24; font-size: 0.9rem; text-align: center; font-weight: 500;
        }
        
        .card {
            background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
            padding: var(--spacing); backdrop-filter: blur(12px); box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .card h2 { margin-top: 0; font-size: 1.25rem; font-weight: 600; border-bottom: 1px solid var(--border); padding-bottom: 0.6rem; }
        
        .tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .tabs button {
            flex: 1; padding: 0.65rem; background: rgba(0,0,0,0.2); border: 1px solid var(--border);
            border-radius: var(--radius); color: var(--text-secondary); font-weight: 500; cursor: pointer; transition: all 0.2s;
        }
        .tabs button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
        
        .upload-area {
            border: 2px dashed var(--accent); border-radius: var(--radius); padding: 2rem; text-align: center; cursor: pointer;
            transition: background 0.2s; min-height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: center;
            background: var(--accent-light);
        }
        .upload-area:hover, .upload-area.hover { background: rgba(99, 102, 241, 0.25); }
        
        .preview img { max-width: 100%; max-height: 280px; object-fit: contain; border-radius: var(--radius); border: 1px solid var(--border); margin-top: 0.5rem; }
        
        .canvas-container { position: relative; }
        canvas {
            border: 1px solid var(--border); border-radius: var(--radius); background: #fff; width: 100%; display: block;
        }
        
        .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1rem; }
        button, .btn {
            padding: 0.6rem 1.2rem; background: var(--accent); border: none; color: #fff; font-weight: 500;
            border-radius: var(--radius); cursor: pointer; transition: background 0.2s; font-size: 0.9rem;
            display: inline-flex; align-items: center; justify-content: center; gap: 0.4rem;
        }
        button:disabled { background: #374151; color: #9ca3af; cursor: not-allowed; }
        button:hover:not(:disabled) { background: var(--accent-hover); }
        .btn-secondary { background: rgba(255,255,255,0.1); color: var(--text-primary); }
        .btn-secondary:hover:not(:disabled) { background: rgba(255,255,255,0.2); }
        
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1rem;
        }
        .metric-card {
            background: rgba(0,0,0,0.3); border: 1px solid var(--border); padding: 0.8rem 1rem; border-radius: var(--radius);
        }
        .metric-card .title { font-size: 0.78rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
        .metric-card .value { font-size: 1.25rem; font-weight: 700; color: #a5b4fc; margin-top: 0.2rem; }
        
        .result-box {
            white-space: pre-wrap; background: rgba(0,0,0,0.4); padding: 1rem; border-radius: var(--radius);
            border: 1px solid var(--border); max-height: 220px; overflow-y: auto; font-family: monospace; font-size: 0.95rem; line-height: 1.5;
        }
        
        .audio-player-container {
            background: rgba(0,0,0,0.3); border: 1px solid var(--border); padding: 1rem; border-radius: var(--radius);
            display: flex; flex-direction: column; gap: 0.6rem; margin-top: 1rem;
        }
        audio { width: 100%; border-radius: 8px; outline: none; }
        
        .crops-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.75rem; margin-top: 1rem;
        }
        .crop-item {
            background: rgba(0,0,0,0.3); border: 1px solid var(--border); padding: 0.5rem; border-radius: 8px; text-align: center;
        }
        .crop-item img { max-width: 100%; max-height: 80px; object-fit: contain; border-radius: 4px; background: #fff; }
        .crop-item span { display: block; font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.3rem; }
    </style>
</head>
<body>
    <div class='container'>
        <div id='server-banner'>⏳ Server is loading AI models (PaddleOCR & TrOCR). Please wait…</div>
        
        <header>
            <h1>Handwritten Text Recognition Dashboard</h1>
            <p>End-to-End OCR, Handwriting Transcription, LLM Correction & Speech Synthesis</p>
        </header>

        <div class='card'>
            <h2>Input Source</h2>
            <div class='tabs'>
                <button id='tab-upload' class='active'>📁 Upload Image</button>
                <button id='tab-whiteboard'>✏️ Interactive Whiteboard</button>
            </div>
            
            <div id='mode-upload' class='mode'>
                <div id='upload-area' class='upload-area'>
                    <p style='font-size:1.5rem; margin:0;'>📄</p>
                    <p style='font-weight:600; margin:0.4rem 0 0.2rem;'>Drag & drop handwritten image here</p>
                    <p style='color:var(--text-secondary); font-size:0.85rem; margin-bottom:0.8rem;'>Supports PNG, JPG, JPEG, WEBP</p>
                    <button id='choose-btn'>Choose File</button>
                </div>
                <input type='file' id='file-input' accept='image/*' style='display:none;'>
                
                <div id='image-preview' class='preview' style='display:none;'>
                    <p><strong>Input Preview</strong></p>
                    <img id='preview-img' src=''>
                    <p id='preview-info' style='color:var(--text-secondary); font-size:0.85rem;'></p>
                    <div class='actions'>
                        <button id='replace-btn' class='btn-secondary'>Replace Image</button>
                        <button id='remove-btn' class='btn-secondary'>Remove</button>
                    </div>
                </div>
            </div>

            <div id='mode-whiteboard' class='mode' style='display:none;'>
                <div class='canvas-container'>
                    <canvas id='whiteboard'></canvas>
                </div>
                <div class='actions'>
                    <button id='undo-btn' class='btn-secondary'>Undo</button>
                    <button id='redo-btn' class='btn-secondary'>Redo</button>
                    <button id='clear-canvas-btn' class='btn-secondary'>Clear Canvas</button>
                </div>
            </div>

            <div class='actions' style='margin-top: 1.2rem;'>
                <button id='recognize-btn' style='font-size:1rem; padding:0.75rem 1.5rem;' disabled>🚀 Process Handwriting Pipeline</button>
            </div>

            <div id='loading' style='display:none; margin-top: 1rem;'>
                <p id='loading-msg' style='color:#a5b4fc; font-weight:500;'>⏳ Running detection, recognition, and LLM understanding…</p>
            </div>
            <div id='error-msg' style='color:#f87171; margin-top: 1rem; display:none; background:rgba(248,113,113,0.1); padding:0.75rem; border-radius:8px; border:1px solid #f87171;'></div>
        </div>

        <div class='card'>
            <h2>Recognition Results & Pipeline Metrics</h2>
            
            <div id='result-section' style='display:none;'>
                <div class='metrics-grid'>
                    <div class='metric-card'>
                        <div class='title'>Lines Detected</div>
                        <div id='metric-lines' class='value'>0</div>
                    </div>
                    <div class='metric-card'>
                        <div class='title'>Processing Time</div>
                        <div id='metric-time' class='value'>0.0s</div>
                    </div>
                    <div class='metric-card'>
                        <div class='title'>HTR Engine</div>
                        <div class='value' style='font-size:1rem; color:#818cf8;'>TrOCR Base</div>
                    </div>
                    <div class='metric-card'>
                        <div class='title'>LLM Model</div>
                        <div id='metric-llm' class='value' style='font-size:1rem; color:#34d399;'>Gemma 2B</div>
                    </div>
                </div>

                <h3>RAW HTR TRANSCRIPTION</h3>
                <div id='raw-text' class='result-box'></div>

                <h3 style='margin-top:1.2rem;'>FINAL CORRECTED TEXT (Gemma LLM)</h3>
                <div id='final-text' class='result-box' style='border-color: #6366f1;'></div>

                <div class='actions'>
                    <button id='copy-btn'>📋 Copy Final Text</button>
                    <button id='download-btn' class='btn-secondary'>💾 Download TXT</button>
                    <button id='clear-result-btn' class='btn-secondary'>Clear</button>
                </div>

                <div class='audio-player-container'>
                    <div style='font-weight:600; display:flex; align-items:center; gap:0.4rem;'>
                        🔊 Audio Speech Synthesis (TTS)
                    </div>
                    <audio id='audio-player' controls></audio>
                    <div>
                        <a id='download-audio-link' href='/audio' download='speech.wav' class='btn btn-secondary' style='text-decoration:none; font-size:0.82rem; padding:0.4rem 0.8rem;'>🎵 Download Speech WAV</a>
                    </div>
                </div>

                <div style='margin-top: 1.5rem;'>
                    <h3>VISUAL DETECTION & LINE CROPS</h3>
                    <div style='margin-bottom: 0.8rem;'>
                        <img id='bboxes-img' style='max-width:100%; border-radius:8px; border:1px solid var(--border); display:none;'>
                    </div>
                    <div id='crops-container' class='crops-grid'></div>
                </div>
            </div>

            <div id='no-result' style='text-align:center; padding: 2rem 1rem; color: var(--text-secondary);'>
                <p style='font-size:1.1rem; margin-bottom:0.4rem;'>No recognition output available yet.</p>
                <p style='font-size:0.88rem; margin:0;'>Upload an image or draw on the whiteboard above and click <strong>Process Handwriting Pipeline</strong>.</p>
            </div>
        </div>
    </div>

    <script>
        const tabUpload = document.getElementById('tab-upload');
        const tabWhiteboard = document.getElementById('tab-whiteboard');
        const modeUpload = document.getElementById('mode-upload');
        const modeWhiteboard = document.getElementById('mode-whiteboard');
        
        function setActiveTab(tab) {
            if (tab === 'upload') {
                tabUpload.classList.add('active'); tabWhiteboard.classList.remove('active');
                modeUpload.style.display = 'block'; modeWhiteboard.style.display = 'none';
            } else {
                tabWhiteboard.classList.add('active'); tabUpload.classList.remove('active');
                modeWhiteboard.style.display = 'block'; modeUpload.style.display = 'none';
            }
            resetState();
        }
        tabUpload.onclick = () => setActiveTab('upload');
        tabWhiteboard.onclick = () => setActiveTab('whiteboard');

        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const chooseBtn = document.getElementById('choose-btn');
        const previewDiv = document.getElementById('image-preview');
        const previewImg = document.getElementById('preview-img');
        const previewInfo = document.getElementById('preview-info');
        const replaceBtn = document.getElementById('replace-btn');
        const removeBtn = document.getElementById('remove-btn');
        const recognizeBtn = document.getElementById('recognize-btn');
        let selectedFile = null;

        chooseBtn.onclick = () => fileInput.click();
        replaceBtn.onclick = () => fileInput.click();
        removeBtn.onclick = () => { selectedFile = null; previewDiv.style.display='none'; recognizeBtn.disabled = true; };
        fileInput.onchange = (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); };

        ['dragenter','dragover','dragleave','drop'].forEach(ev => uploadArea.addEventListener(ev, e => e.preventDefault()));
        uploadArea.addEventListener('dragenter', () => uploadArea.classList.add('hover'));
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('hover'));
        uploadArea.addEventListener('drop', e => {
            uploadArea.classList.remove('hover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) { showError('File must be an image.'); return; }
            selectedFile = file;
            previewImg.src = URL.createObjectURL(file);
            previewInfo.textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
            previewDiv.style.display = 'block';
            recognizeBtn.disabled = false;
        }

        const canvas = document.getElementById('whiteboard');
        const ctx = canvas.getContext('2d');
        let drawing = false;
        const undoStack = [], redoStack = [];
        
        function pushState() { undoStack.push(canvas.toDataURL()); if (undoStack.length>20) undoStack.shift(); }
        
        canvas.addEventListener('pointerdown', e => { drawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); });
        canvas.addEventListener('pointermove', e => {
            if (!drawing) return;
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.strokeStyle = '#000000'; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.stroke();
        });
        
        const canvasContainer = canvas.parentElement;
        function resizeCanvas() {
            const newW = canvasContainer.clientWidth || 500;
            const newH = Math.max(220, Math.round(newW * 0.45));
            const snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
            canvas.width = newW; canvas.height = newH;
            ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.putImageData(snapshot, 0, 0);
        }
        resizeCanvas();
        if (window.ResizeObserver) new ResizeObserver(resizeCanvas).observe(canvasContainer);

        canvas.addEventListener('pointerup', () => { if (drawing) { drawing = false; pushState(); redoStack.length = 0; recognizeBtn.disabled = false; } });
        document.getElementById('clear-canvas-btn').onclick = () => { ctx.fillStyle="#ffffff"; ctx.fillRect(0,0,canvas.width,canvas.height); pushState(); };
        document.getElementById('undo-btn').onclick = () => {
            if (undoStack.length > 0) { redoStack.push(canvas.toDataURL()); const imgData = undoStack.pop(); const img = new Image(); img.onload = () => { ctx.drawImage(img,0,0); }; img.src = imgData; }
        };
        document.getElementById('redo-btn').onclick = () => {
            if (redoStack.length > 0) { undoStack.push(canvas.toDataURL()); const imgData = redoStack.pop(); const img = new Image(); img.onload = () => { ctx.drawImage(img,0,0); }; img.src = imgData; }
        };

        const loadingDiv = document.getElementById('loading');
        const errorDiv = document.getElementById('error-msg');
        const resultSection = document.getElementById('result-section');
        const rawTextDiv = document.getElementById('raw-text');
        const finalTextDiv = document.getElementById('final-text');
        const noResultDiv = document.getElementById('no-result');
        const audioPlayer = document.getElementById('audio-player');
        const bboxesImg = document.getElementById('bboxes-img');
        const cropsContainer = document.getElementById('crops-container');

        recognizeBtn.onclick = async () => {
            errorDiv.style.display = 'none';
            loadingDiv.style.display = 'block';
            recognizeBtn.disabled = true;
            
            const formData = new FormData();
            if (modeUpload.style.display !== 'none' && selectedFile) {
                formData.append('file', selectedFile);
                await sendRequest(formData);
            } else if (modeWhiteboard.style.display !== 'none') {
                canvas.toBlob(blob => {
                    formData.append('file', blob, 'whiteboard.png');
                    sendRequest(formData);
                }, 'image/png');
            }
        };

        async function sendRequest(formData) {
            try {
                const resp = await fetch('/process', { method: 'POST', body: formData });
                const data = await resp.json();
                
                if (resp.ok) {
                    if (data.message && !data.raw_text) {
                        showError(data.message);
                    } else {
                        rawTextDiv.textContent = data.raw_text || '(No text detected)';
                        finalTextDiv.textContent = data.final_text || '(No text produced)';
                        
                        document.getElementById('metric-lines').textContent = data.stats ? data.stats.regions_detected : '0';
                        document.getElementById('metric-time').textContent = data.stats ? data.stats.processing_time : '0.0s';
                        document.getElementById('metric-llm').textContent = data.stats ? data.stats.llm_model : 'Gemma';
                        
                        audioPlayer.src = '/audio?t=' + new Date().getTime();
                        
                        if (data.detected_boxes_b64) {
                            bboxesImg.src = 'data:image/png;base64,' + data.detected_boxes_b64;
                            bboxesImg.style.display = 'block';
                        } else {
                            bboxesImg.style.display = 'none';
                        }

                        cropsContainer.innerHTML = '';
                        if (data.crops && data.crops.length > 0) {
                            data.crops.forEach((c_b64, idx) => {
                                const item = document.createElement('div');
                                item.className = 'crop-item';
                                item.innerHTML = `<img src="data:image/png;base64,${c_b64}"><span>Line Crop ${idx}</span>`;
                                cropsContainer.appendChild(item);
                            });
                        }

                        resultSection.style.display = 'block';
                        noResultDiv.style.display = 'none';
                    }
                } else {
                    showError(data.detail || 'Recognition process failed.');
                }
            } catch (e) {
                showError('Error connecting to the pipeline server.');
            } finally {
                loadingDiv.style.display = 'none';
                recognizeBtn.disabled = false;
            }
        }

        function showError(msg) { errorDiv.textContent = msg; errorDiv.style.display = 'block'; }

        document.getElementById('copy-btn').onclick = async () => {
            try { await navigator.clipboard.writeText(finalTextDiv.textContent); alert('Final text copied to clipboard ✓'); } catch(e){ alert('Copy failed'); }
        };
        document.getElementById('download-btn').onclick = () => {
            const blob = new Blob([finalTextDiv.textContent], {type:'text/plain'});
            const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'result.txt'; a.click();
        };
        document.getElementById('clear-result-btn').onclick = () => {
            resultSection.style.display = 'none'; noResultDiv.style.display = 'block';
        };

        function resetState() {
            errorDiv.style.display='none'; loadingDiv.style.display='none'; recognizeBtn.disabled = true;
            selectedFile = null; previewDiv.style.display='none';
            ctx.fillStyle="#ffffff"; ctx.fillRect(0,0,canvas.width,canvas.height);
        }

        (function pollHealth() {
            fetch('/health').then(r => r.json()).then(d => {
                const banner = document.getElementById('server-banner');
                if (!d.models_ready) { banner.style.display = 'block'; setTimeout(pollHealth, 3000); }
                else { banner.style.display = 'none'; }
            }).catch(() => setTimeout(pollHealth, 5000));
        })();
    </script>
</body>
</html>
"""

@app.get('/', response_class=HTMLResponse)
async def get_root():
    return HTML_PAGE

@app.post('/process')
def process_image(file: UploadFile = File(...)):
    start_time = time.time()

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Uploaded file is not an image')

    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    total_bytes = 0
    try:
        with open(temp_path, 'wb') as out_file:
            while True:
                chunk = file.file.read(65536)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f'File size exceeds limit of {MAX_UPLOAD_BYTES // (1024*1024)} MB.'
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save upload: {e}')
    finally:
        file.file.close()

    try:
        with PilImage.open(temp_path) as img:
            img.verify()
    except Exception:
        try:
            if os.path.exists(temp_path): os.remove(temp_path)
        except Exception: pass
        raise HTTPException(status_code=400, detail='Uploaded file is corrupted or not a valid image.')

    try:
        detected_regions = detect_text(temp_path, debug=True, merge_lines=True)

        if not detected_regions:
            return JSONResponse(
                status_code=200,
                content={
                    "raw_text": "",
                    "final_text": "",
                    "message": "No text regions were detected in the image.",
                    "stats": {"regions_detected": 0, "processing_time": f"{time.time() - start_time:.2f}s"}
                }
            )

        crops = [r['crop'] for r in detected_regions]
        raw_lines = recognize_batch(crops)
        raw_text = '\n'.join([line for line in raw_lines if line.strip()])

        gemma_model = get_available_model()
        corrected_text, final_text = process_text_pipeline(raw_text, model_name=gemma_model, debug=False)

        audio_file = os.path.join("output", "speech.wav")
        def _speak_bg(txt):
            try:
                speak(txt, output_wav_path=audio_file)
            except Exception:
                pass

        if final_text:
            threading.Thread(target=_speak_bg, args=(final_text,), daemon=True).start()

        boxes_b64 = ""
        bbox_path = os.path.join("output", "detected_boxes.png")
        if os.path.exists(bbox_path):
            with open(bbox_path, "rb") as f:
                boxes_b64 = base64.b64encode(f.read()).decode('utf-8')

        crops_b64 = []
        for reg in detected_regions:
            c_path = reg.get('crop_path')
            if c_path and os.path.exists(c_path):
                with open(c_path, "rb") as f:
                    crops_b64.append(base64.b64encode(f.read()).decode('utf-8'))

        duration = time.time() - start_time

        return JSONResponse(content={
            "raw_text": raw_text,
            "final_text": final_text,
            "detected_boxes_b64": boxes_b64,
            "crops": crops_b64,
            "stats": {
                "regions_detected": len(detected_regions),
                "processing_time": f"{duration:.2f}s",
                "htr_model": "microsoft/trocr-base-handwritten",
                "llm_model": gemma_model
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
