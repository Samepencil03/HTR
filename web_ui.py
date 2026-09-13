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
        body {font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #1e1e2f, #2a2a40); color:#f0f0f0; display:flex; align-items:center; justify-content:center; height:100vh; margin:0;}
        .container {background: rgba(255,255,255,0.08); padding:2rem; border-radius:12px; box-shadow:0 8px 32px rgba(0,0,0,0.2); backdrop-filter:blur(8px); width:400px;}
        h1 {text-align:center; margin-bottom:1rem; font-size:1.5rem;}
        input[type='file'] {display:block; width:100%; margin:1rem 0;}
        button {background:#4e70e2; border:none; padding:0.6rem 1.2rem; color:#fff; border-radius:6px; cursor:pointer; width:100%; font-size:1rem; transition:background 0.2s;}
        button:hover {background:#3b5ac4;}
        pre {background:rgba(0,0,0,0.3); padding:1rem; border-radius:6px; overflow:auto; max-height:200px;}
    </style>
</head>
<body>
<div class='container'>
    <h1>Handwritten Text Recognition</h1>
    <form id='upload-form' enctype='multipart/form-data'>
        <input type='file' name='file' accept='image/*' required />
        <button type='submit'>Upload & Process</button>
    </form>
    <pre id='result'></pre>
</div>
<script>
    const form = document.getElementById('upload-form');
    const resultBox = document.getElementById('result');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = form.elements['file'];
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        resultBox.textContent = 'Processing...';
        try {
            const resp = await fetch('/process', {method: 'POST', body: formData});
            const data = await resp.json();
            if (resp.ok) {
                resultBox.textContent = `RAW TEXT:\n${data.raw_text}\n\nFINAL TEXT:\n${data.final_text}`;
            } else {
                resultBox.textContent = `Error: ${data.detail || 'unknown'}`;
            }
        } catch (err) {
            resultBox.textContent = 'Network error: ' + err.message;
        }
    });
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
