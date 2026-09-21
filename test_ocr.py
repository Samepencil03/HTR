"""
Quick smoke-test for PaddleOCR detection.

Usage:
    python test_ocr.py path/to/image.png
    python test_ocr.py                        # uses create_sample_image.py to generate one
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Smoke-test PaddleOCR on an image.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the image to test. If omitted, a sample image is generated automatically.",
    )
    args = parser.parse_args()

    # FIX #10: no hardcoded absolute path — generate a sample if none supplied
    image_path = args.image_path
    if not image_path:
        from create_sample_image import generate_sample_image
        image_path = "input/sample_handwritten.png"
        generate_sample_image(image_path)

    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        raise SystemExit(1)

    from paddleocr import PaddleOCR

    print(f"Running PaddleOCR on: {image_path}")
    ocr = PaddleOCR(lang="en", show_log=False)
    results = ocr.predict(image_path)

    for result in results:
        result.print()

if __name__ == "__main__":
    main()
