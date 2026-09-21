import os
import sys
import subprocess

def speak(text: str, output_wav_path: str = "output/speech.wav") -> bool:
    """
    Speak text using a local Linux-compatible TTS engine and export an audio speech file (output/speech.wav).
    Supports pyttsx3, espeak-ng / espeak WAV export, and spd-say playback.
    """
    if not text or not text.strip():
        return False

    output_dir = os.path.dirname(output_wav_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    saved = False

    # 1. Try pyttsx3 save to WAV file & live playback
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, output_wav_path)
        engine.runAndWait()
        if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
            print(f"[TTS Info] Saved audio speech output to '{output_wav_path}'.")
            saved = True
        
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        return True
    except Exception as e:
        pass

    # 2. Fallback: espeak-ng / espeak WAV export
    for cmd in ["espeak-ng", "espeak"]:
        try:
            res = subprocess.run([cmd, "-w", output_wav_path, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if res.returncode == 0 and os.path.exists(output_wav_path):
                print(f"[TTS Info] Saved audio speech output to '{output_wav_path}' via {cmd}.")
                saved = True
                for player in ["aplay", "paplay", "ffplay"]:
                    try:
                        subprocess.run([player, output_wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                        break
                    except Exception:
                        pass
                return True
        except Exception:
            pass

    # 3. Fallback: spd-say
    try:
        res = subprocess.run(["spd-say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        if res.returncode == 0:
            return True
    except Exception:
        pass

    print("[TTS Info] Text-to-Speech output:")
    print(f"\"{text}\"")
    return saved
