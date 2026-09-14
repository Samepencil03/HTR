import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import pipeline functions
from detector import detect_text
from htr import recognize_text
from llm import process_text_pipeline
from tts import speak

app = FastAPI(title="Handwritten Text Recognition Web UI")

# Allow all origins for local testing (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PAGE = """
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <title>Handwritten Text Recognition</title>
    <style>
        :root {
            --bg-primary: #1e1e2f;
            --bg-card: rgba(255,255,255,0.08);
            --accent: #4e70e2;
            --text: #f0f0f0;
            --border-radius: 12px;
            --spacing: 1rem;
        }
        body {
            margin:0; padding:0; font-family: 'Inter', sans-serif; background: var(--bg-primary); color: var(--text);
            display:flex; align-items:center; justify-content:center; min-height:100vh;
        }
        .container {
            background: var(--bg-card); padding: var(--spacing); border-radius: var(--border-radius);
            box-shadow:0 8px 32px rgba(0,0,0,0.2); backdrop-filter:blur(8px); width:100%; max-width:960px;
            display:flex; flex-direction:column; gap: var(--spacing);
        }
        header { text-align:center; }
        header h1 { margin:0; font-size:2rem; }
        header p { margin:0.2rem 0 0; opacity:0.8; }
        .card {
            background: var(--bg-card); padding: var(--spacing); border-radius: var(--border-radius);
            box-shadow:0 4px 16px rgba(0,0,0,0.15);
        }
        .tabs { display:flex; gap:0.5rem; margin-bottom: var(--spacing); }
        .tabs button {
            flex:1; padding:0.6rem 0; background: var(--bg-primary); border:none; border-radius: var(--border-radius);
            color: var(--text); cursor:pointer; transition:background 0.2s;
        }
        .tabs button.active { background: var(--accent); }
        .upload-area {
            border: 2px dashed var(--accent); border-radius: var(--border-radius); padding: var(--spacing); text-align:center; cursor:pointer;
            transition:background 0.2s; min-height:200px; display:flex; flex-direction:column; align-items:center; justify-content:center;
        }
        .upload-area.hover { background: rgba(78,112,226,0.2); }
        .preview {
            margin-top: var(--spacing); text-align:center;
        }
        .preview img { max-width:100%; max-height:300px; object-fit:contain; border-radius: var(--border-radius); }
        .canvas-container { position:relative; }
        canvas { border:1px solid var(--accent); border-radius: var(--border-radius); background:#fff; }
        .actions { display:flex; gap:0.5rem; flex-wrap:wrap; }
        button, .actions button { padding:0.5rem 1rem; background: var(--accent); border:none; color: #fff; border-radius: var(--border-radius); cursor:pointer; transition:background 0.2s; }
        button:disabled, .actions button:disabled { background: #555; cursor: not-allowed; }
        button:hover:not(:disabled), .actions button:hover:not(:disabled) { background:#3b5ac4; }
        .result-box { white-space:pre-wrap; background:rgba(0,0,0,0.3); padding:var(--spacing); border-radius: var(--border-radius); max-height:200px; overflow:auto; }
        @media (max-width: 768px) {
            .flex-row { flex-direction:column; }
        }
        .flex-row { display:flex; gap: var(--spacing); }
        .flex-col { flex:1; }
    </style>
</head>
<body>
    <div class='container'>
        <header>
            <h1>Handwritten Text Recognition</h1>
            <p>Convert handwritten content into clear digital text</p>
        </header>
        <div class='card'>
            <h2>Input Source</h2>
            <div class='tabs'>
                <button id='tab-upload' class='active'>Upload Image</button>
                <button id='tab-whiteboard'>Whiteboard</button>
            </div>
            <div id='mode-upload' class='mode'>
                <div id='upload-area' class='upload-area'>
                    <p>⬆️ Upload your image</p>
                    <p>Drag & drop an image here</p>
                    <p>or</p>
                    <button id='choose-btn'>Choose Image</button>
                    <p>Supported: PNG, JPG, JPEG, WEBP</p>
                </div>
                <input type='file' id='file-input' accept='image/*' style='display:none;'>
                <div id='image-preview' class='preview' style='display:none;'>
                    <p><strong>Preview</strong></p>
                    <img id='preview-img' src=''>
                    <p id='preview-info'></p>
                    <div class='actions'>
                        <button id='replace-btn'>Replace Image</button>
                        <button id='remove-btn'>Remove Image</button>
                    </div>
                </div>
            </div>
            <div id='mode-whiteboard' class='mode' style='display:none;'>
                <div class='canvas-container'>
                    <canvas id='whiteboard' width='500' height='300'></canvas>
                </div>
                <div class='actions'>
                    <button id='undo-btn'>Undo</button>
                    <button id='redo-btn'>Redo</button>
                    <button id='clear-canvas-btn'>Clear</button>
                </div>
            </div>
            <div class='actions' style='margin-top: var(--spacing);'>
                <button id='recognize-btn' disabled>Recognize Handwriting</button>
            </div>
            <div id='loading' style='display:none; margin-top: var(--spacing);'>
                <p>⏳ Recognizing...</p>
            </div>
            <div id='error-msg' style='color:#ff6b6b; margin-top: var(--spacing); display:none;'></div>
        </div>
        <div class='card'>
            <h2>Recognition Result</h2>
            <div id='result-section' style='display:none;'>
                <h3>RAW TEXT</h3>
                <div id='raw-text' class='result-box'></div>
                <h3>FINAL TEXT</h3>
                <div id='final-text' class='result-box'></div>
                <div class='actions' style='margin-top: var(--spacing);'>
                    <button id='copy-btn'>Copy</button>
                    <button id='download-btn'>Download</button>
                    <button id='clear-result-btn'>Clear Result</button>
                </div>
            </div>
            <div id='no-result' style='text-align:center; opacity:0.7;'>
                <p>No recognition result yet.</p>
                <p>Submit an image or whiteboard drawing to see the result.</p>
            </div>
        </div>
    </div>
    <script>
        // Tab handling
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

        // Upload handling
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const chooseBtn = document.getElementById('choose-btn');
        const previewDiv = document.getElementById('image-preview');
        const previewImg = document.getElementById('preview-img');
        const previewInfo = document.getElementById('preview-info');
        const replaceBtn = document.getElementById('replace-btn');
        const removeBtn = document.getElementById('remove-btn');
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
            const dt = e.dataTransfer;
            if (dt.files && dt.files[0]) handleFile(dt.files[0]);
        });
        function handleFile(file) {
            if (!file.type.startsWith('image/')) { showError('File must be an image.'); return; }
            selectedFile = file;
            const url = URL.createObjectURL(file);
            previewImg.src = url;
            previewInfo.textContent = `${file.name}`;
            previewDiv.style.display = 'block';
            recognizeBtn.disabled = false;
        }

        // Whiteboard handling
        const canvas = document.getElementById('whiteboard');
        const ctx = canvas.getContext('2d');
        let drawing = false;
        const undoStack = [];
        const redoStack = [];
        function pushState() { undoStack.push(canvas.toDataURL()); if (undoStack.length>20) undoStack.shift(); }
        canvas.addEventListener('pointerdown', e => { drawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); });
        canvas.addEventListener('pointermove', e => { if (!drawing) return; ctx.lineTo(e.offsetX, e.offsetY); ctx.strokeStyle = '#4e70e2'; ctx.lineWidth = 2; ctx.stroke(); });
        canvas.addEventListener('pointerup', () => { if (drawing) { drawing = false; pushState(); redoStack.length = 0; } });
        canvas.addEventListener('pointerout', () => { if (drawing) { drawing = false; pushState(); redoStack.length = 0; } });
        document.getElementById('clear-canvas-btn').onclick = () => { ctx.clearRect(0,0,canvas.width,canvas.height); pushState(); };
        document.getElementById('undo-btn').onclick = () => {
            if (undoStack.length > 0) { redoStack.push(canvas.toDataURL()); const imgData = undoStack.pop(); const img = new Image(); img.onload = () => { ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(img,0,0); }; img.src = imgData; }
        };
        document.getElementById('redo-btn').onclick = () => {
            if (redoStack.length > 0) { undoStack.push(canvas.toDataURL()); const imgData = redoStack.pop(); const img = new Image(); img.onload = () => { ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(img,0,0); }; img.src = imgData; }
        };

        // Recognition handling
        const recognizeBtn = document.getElementById('recognize-btn');
        const loadingDiv = document.getElementById('loading');
        const errorDiv = document.getElementById('error-msg');
        const resultSection = document.getElementById('result-section');
        const rawTextDiv = document.getElementById('raw-text');
        const finalTextDiv = document.getElementById('final-text');
        const noResultDiv = document.getElementById('no-result');
        recognizeBtn.onclick = async () => {
            errorDiv.style.display='none';
            loadingDiv.style.display='block';
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
            } else {
                showError('Please provide an image or drawing first.');
                loadingDiv.style.display='none';
                recognizeBtn.disabled = false;
            }
        };
        async function sendRequest(formData) {
            try {
                const resp = await fetch('/process', {method:'POST', body:formData});
                const data = await resp.json();
                if (resp.ok) {
                    rawTextDiv.textContent = data.raw_text || '';
                    finalTextDiv.textContent = data.final_text || '';
                    resultSection.style.display = 'block';
                    noResultDiv.style.display = 'none';
                } else {
                    showError(data.detail || 'Recognition failed.');
                }
            } catch (e) {
                showError('Unable to connect to the recognition server.');
            } finally {
                loadingDiv.style.display='none';
                recognizeBtn.disabled = false;
            }
        }
        function showError(msg) { errorDiv.textContent = msg; errorDiv.style.display='block'; }

        // Result actions
        document.getElementById('copy-btn').onclick = async () => {
            try { await navigator.clipboard.writeText(finalTextDiv.textContent); alert('Copied ✓'); } catch(e){ alert('Copy failed'); }
        };
        document.getElementById('download-btn').onclick = () => {
            const blob = new Blob([finalTextDiv.textContent], {type:'text/plain'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a'); a.href = url; a.download = 'recognition_result.txt'; a.click(); URL.revokeObjectURL(url);
        };
        document.getElementById('clear-result-btn').onclick = () => {
            rawTextDiv.textContent = '';
            finalTextDiv.textContent = '';
            resultSection.style.display='none';
            noResultDiv.style.display='block';
        };

        function resetState() {
            errorDiv.style.display='none';
            loadingDiv.style.display='none';
            recognizeBtn.disabled = true;
            selectedFile = null;
            previewDiv.style.display='none';
            ctx.clearRect(0,0,canvas.width,canvas.height);
            undoStack.length = 0; redoStack.length = 0;
        }
    </script>
</body>
</html>
"""

@app.get('/', response_class=HTMLResponse)
async def get_root():
    return HTML_PAGE

@app.post('/process')
async def process_image(file: UploadFile = File(...)):
    # Validate mime type (basic check)
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Uploaded file is not an image')
    # Save to a temporary location
    temp_dir = 'tmp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{file.filename}")
    try:
        with open(temp_path, 'wb') as out_file:
            shutil.copyfileobj(file.file, out_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save upload: {e}')
    finally:
        await file.close()

    try:
        # Stage 1: Text detection
        detected_regions = detect_text(temp_path, debug=False)
        if not detected_regions:
            raise HTTPException(status_code=200, detail='No text regions detected')
        # Stage 2: Handwritten Text Recognition
        raw_lines = []
        for region in detected_regions:
            crop_img = region['crop']
            line_text = recognize_text(crop_img)
            if line_text:
                raw_lines.append(line_text)
        raw_text = '\n'.join(raw_lines)
        # Stage 3: LLM correction & understanding
        corrected_text, final_text = process_text_pipeline(raw_text, model_name=None, debug=False)
        # Optional: Speak the final text (non‑blocking)
        try:
            speak(final_text)
        except Exception:
            pass
        return JSONResponse(content={"raw_text": raw_text, "final_text": final_text})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except Exception:
            pass
