import sys
import subprocess

def speak(text: str):
    """
    Speak text using a local Linux-compatible offline TTS engine (pyttsx3 / espeak-ng / espeak / spd-say).
    """
    if not text or not text.strip():
        return

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        return
    except Exception as e:
        print(f"[TTS Info] pyttsx3 audio engine unavailable ({e}). Trying system CLI fallbacks...")

    for cmd in ["espeak-ng", "espeak", "spd-say"]:
        try:
            res = subprocess.run([cmd, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if res.returncode == 0:
                return
        except Exception:
            pass

    print("[TTS Info] Offline audio playback unavailable on current terminal environment.")
    print(f"[TTS Spoken Text] \"{text}\"")
