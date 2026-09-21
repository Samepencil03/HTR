import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Handwritten Text Recognition Pipeline")
    parser.add_argument("image_path", type=str, help="Path to input handwritten text image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save line crops and output bounding boxes")
    parser.add_argument("--model", type=str, default=None, help="Name of local Gemma model in Ollama (e.g. gemma2:2b)")
    return parser.parse_args()

def main():
    args = parse_args()
    image_path = args.image_path
    debug = args.debug

    # Stage 1: Loading image
    print("[1/5] Loading image...")
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at '{image_path}'")
        sys.exit(1)

    # Stage 2: Detecting text regions with PaddleOCR
    print("[2/5] Detecting text regions with PaddleOCR...")
    try:
        from detector import detect_text
        detected_regions = detect_text(image_path, debug=debug, merge_lines=True)
    except Exception as e:
        print(f"Error during PaddleOCR text detection stage: {e}")
        sys.exit(1)

    if not detected_regions:
        print("Warning: No text regions detected in the image.")
        raw_text = ""
        final_text = ""
    else:
        # Stage 3: Running handwritten text recognition (HTR)
        print(f"[3/5] Running handwritten text recognition on {len(detected_regions)} line regions...")
        try:
            from htr import recognize_batch
            crops = [r['crop'] for r in detected_regions]
            raw_lines = recognize_batch(crops)

            if debug:
                for idx, (region, text) in enumerate(zip(detected_regions, raw_lines)):
                    box = region['box']
                    print(f"[DEBUG] Line {idx} Box={box} HTR raw output: '{text}'")

            valid_lines = [line for line in raw_lines if line.strip()]
            raw_text = "\n".join(valid_lines)
        except Exception as e:
            print(f"Error during HTR stage initialization: {e}")
            sys.exit(1)

    # Stage 4: Correcting and understanding text with Gemma
    print("[4/5] Correcting and understanding text with Gemma...")
    corrected_text = ""
    final_text = ""
    if raw_text:
        try:
            from llm import process_text_pipeline
            corrected_text, final_text = process_text_pipeline(raw_text, model_name=args.model, debug=debug)
        except Exception as e:
            print(f"[LLM Warning] Gemma model processing failed ({e}). Using raw HTR transcription.")
            final_text = raw_text
    else:
        final_text = raw_text

    # Save final result to output/result.txt
    os.makedirs("output", exist_ok=True)
    out_file = os.path.join("output", "result.txt")
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write((final_text.strip() if final_text else "") + "\n")
    except Exception as e:
        print(f"Warning: Could not save result to '{out_file}': {e}")

    # Stage 5: Speaking final text
    print("[5/5] Speaking final text...")
    audio_file = os.path.join("output", "speech.wav")
    if final_text:
        try:
            from tts import speak
            speak(final_text, output_wav_path=audio_file)
        except Exception as e:
            print(f"[TTS Warning] Text-to-Speech execution encountered issue: {e}")

    # Print required terminal outputs
    print("\nRAW TEXT:")
    print("----------------------------------------")
    print(raw_text if raw_text else "(No text detected)")
    print("----------------------------------------")

    print("\nFINAL TEXT:")
    print("----------------------------------------")
    print(final_text if final_text else "(No text produced)")
    print("----------------------------------------")
    print(f"\nResult saved to: {out_file}")
    if os.path.exists(audio_file):
        print(f"Audio output saved to: {audio_file}")
    print()

if __name__ == "__main__":
    main()
