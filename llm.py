import os
import requests
import json
import subprocess
import time

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def ensure_ollama_running():
    """
    Check if Ollama server is active on host. If not, attempt to start 'ollama serve' in background.
    """
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if resp.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass

    print("[LLM Warning] Ollama server is not responding. Attempting to start 'ollama serve'...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(12):
            time.sleep(0.5)
            try:
                resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
                if resp.status_code == 200:
                    print("[LLM Info] Ollama server started successfully.")
                    return True
            except requests.exceptions.RequestException:
                pass
    except Exception as e:
        print(f"[LLM Warning] Failed to start 'ollama serve': {e}")

    return False

def get_available_model(requested_model: str = None) -> str:
    """
    Detect available local Gemma model or select user-requested model.
    Prefers lightweight models that fit within available system memory.
    """
    if requested_model:
        return requested_model

    env_model = os.environ.get("GEMMA_MODEL")
    if env_model:
        return env_model

    if not ensure_ollama_running():
        raise RuntimeError("Ollama server is not available at http://localhost:11434. Please start Ollama service.")

    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            if not models:
                raise RuntimeError("No local Ollama models found. Please pull a model using 'ollama pull gemma2:2b'.")

            # Sort models ascending by byte size so smaller models are preferred for RAM compatibility
            models_sorted = sorted(models, key=lambda x: x.get("size", 0))

            # Look for gemma models in sorted order (smallest first)
            for m in models_sorted:
                name = m.get("name", "")
                if "gemma" in name.lower():
                    return name

            # Return smallest model available
            return models_sorted[0].get("name", "")
    except Exception as e:
        raise RuntimeError(f"Error connecting to Ollama: {e}") from e

    return "gemma2:2b"

def query_ollama(prompt: str, model_name: str, system_prompt: str = "") -> str:
    """
    Send generation prompt to Ollama REST API.
    """
    ensure_ollama_running()
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=180)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama request failed (HTTP {response.status_code}): {response.text}")
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama connection error: {e}") from e

def correct_text(raw_text: str, model_name: str = None) -> str:
    """
    Stage A: Correction.
    Fixes obvious HTR/OCR errors (spelling, punctuation, spacing, misread characters)
    without hallucinating facts or altering sentence structure/meaning.
    """
    if not raw_text or not raw_text.strip():
        return ""

    model = get_available_model(model_name)

    system_prompt = (
        "You are an expert handwriting OCR correction assistant.\n"
        "Your task is to correct spelling, punctuation, spacing, and character recognition errors "
        "in raw handwritten OCR transcripts.\n"
        "Rules:\n"
        "1. Fix obvious spelling and character mistakes.\n"
        "2. Do NOT invent new facts or details.\n"
        "3. Do NOT rewrite the author's meaning or sentence structure.\n"
        "4. Output ONLY the corrected transcript text without any introductory remarks, notes, or headers."
    )

    prompt = (
        f"Correct the following raw handwritten OCR transcript:\n\n"
        f"--- RAW TRANSCRIPT ---\n"
        f"{raw_text}\n"
        f"--- END TRANSCRIPT ---\n\n"
        f"Corrected text:"
    )

    return query_ollama(prompt, model, system_prompt=system_prompt)

def understand_text(corrected_text: str, model_name: str = None) -> str:
    """
    Stage B: Understanding.
    Produces a cleaned, polished final representation while preserving original meaning.
    """
    if not corrected_text or not corrected_text.strip():
        return ""

    model = get_available_model(model_name)

    system_prompt = (
        "You are a text understanding and formatting assistant.\n"
        "Format the input text into a clear, clean final representation.\n"
        "Rules:\n"
        "1. Preserve the exact original meaning.\n"
        "2. Do NOT hallucinate missing content or add extra information.\n"
        "3. Output ONLY the final processed text without preamble or explanation."
    )

    prompt = (
        f"Clean and finalize the following corrected transcript:\n\n"
        f"--- CORRECTED TEXT ---\n"
        f"{corrected_text}\n"
        f"--- END TEXT ---\n\n"
        f"Final representation:"
    )

    return query_ollama(prompt, model, system_prompt=system_prompt)

def process_text_pipeline(raw_text: str, model_name: str = None, debug: bool = False) -> tuple[str, str]:
    """
    Executes Stage A (Correction) and Stage B (Understanding) sequentially.
    Returns (corrected_text, final_text).
    """
    model = get_available_model(model_name)
    if debug:
        print(f"[DEBUG] Ollama Model: {model}")

    corrected = correct_text(raw_text, model_name=model)
    if debug:
        print(f"[DEBUG] Stage A (Correction) Output:\n{corrected}\n")

    final = understand_text(corrected, model_name=model)
    if debug:
        print(f"[DEBUG] Stage B (Understanding) Output:\n{final}\n")

    return corrected, final
