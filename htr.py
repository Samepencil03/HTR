import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, RobertaTokenizer

class HTRRecognizer:
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = None
        self.model = None

    def _ensure_loaded(self):
        if self.processor is None or self.model is None:
            try:
                print(f"[HTR Info] Loading HTR model '{self.model_name}' on device '{self.device}'...")
                image_processor = ViTImageProcessor.from_pretrained(self.model_name)
                tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
                self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
                print(f"[HTR Info] HTR model '{self.model_name}' loaded successfully.")
            except Exception as e:
                fallback_name = "microsoft/trocr-small-handwritten"
                if self.model_name != fallback_name:
                    print(f"[HTR Warning] Failed to load '{self.model_name}': {e}. Retrying with '{fallback_name}'...")
                    self.model_name = fallback_name
                    try:
                        image_processor = ViTImageProcessor.from_pretrained(self.model_name)
                        tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
                        self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
                        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
                        print(f"[HTR Info] HTR fallback model '{self.model_name}' loaded successfully.")
                    except Exception as e2:
                        raise RuntimeError(f"HTR model loading failure for both primary and fallback models: {e2}") from e
                else:
                    raise RuntimeError(f"HTR model loading failure for model '{self.model_name}': {e}") from e

    def recognize_text(self, image_crop) -> str:
        """
        Recognize handwritten text from a single cropped image using TrOCR.
        """
        res = self.recognize_batch([image_crop])
        return res[0] if res else ""

    def recognize_batch(self, image_crops: list, max_new_tokens: int = 128) -> list[str]:
        """
        Recognize handwritten text from a list of cropped images in batch using TrOCR.
        """
        if not image_crops:
            return []

        self._ensure_loaded()

        pil_crops = []
        for crop in image_crops:
            if not isinstance(crop, Image.Image):
                pil_crops.append(Image.fromarray(crop).convert("RGB"))
            else:
                pil_crops.append(crop.convert("RGB"))

        try:
            pixel_values = self.processor(images=pil_crops, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return [t.strip() for t in generated_texts]
        except Exception as e:
            print(f"[HTR Warning] Batch recognition failed ({e}). Falling back to per-image processing.")
            results = []
            for crop in pil_crops:
                try:
                    pv = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
                    with torch.no_grad():
                        gids = self.model.generate(pv, max_new_tokens=max_new_tokens)
                    txt = self.processor.batch_decode(gids, skip_special_tokens=True)[0]
                    results.append(txt.strip())
                except Exception as ex:
                    print(f"[HTR Warning] Failed crop recognition: {ex}")
                    results.append("")
            return results

_htr_instance = None

def recognize_text(image_crop, model_name: str = "microsoft/trocr-base-handwritten") -> str:
    global _htr_instance
    if _htr_instance is None:
        _htr_instance = HTRRecognizer(model_name=model_name)
    return _htr_instance.recognize_text(image_crop)

def recognize_batch(image_crops: list, model_name: str = "microsoft/trocr-base-handwritten") -> list[str]:
    global _htr_instance
    if _htr_instance is None:
        _htr_instance = HTRRecognizer(model_name=model_name)
    return _htr_instance.recognize_batch(image_crops)
