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
        Recognize handwritten text from a cropped PIL Image or NumPy array using TrOCR.
        """
        self._ensure_loaded()

        if not isinstance(image_crop, Image.Image):
            image_crop = Image.fromarray(image_crop).convert("RGB")
        else:
            image_crop = image_crop.convert("RGB")

        try:
            pixel_values = self.processor(images=image_crop, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_new_tokens=64)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
        except Exception as e:
            print(f"[HTR Warning] Recognition failed on crop: {e}")
            return ""

_htr_instance = None

def recognize_text(image_crop, model_name: str = "microsoft/trocr-base-handwritten") -> str:
    """
    Module-level function to recognize text using a singleton HTRRecognizer instance.
    """
    global _htr_instance
    if _htr_instance is None:
        _htr_instance = HTRRecognizer(model_name=model_name)
    return _htr_instance.recognize_text(image_crop)
