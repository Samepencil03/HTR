from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

results = ocr.predict(
    "/home/adam/Pictures/Screenshots/test.png"
)

for result in results:
    result.print()
