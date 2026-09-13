import os
import cv2
import numpy as np
from PIL import Image

_detector_instance = None

def get_detector():
    """
    Initialize and return a singleton PaddleOCR text detector instance.
    """
    global _detector_instance
    if _detector_instance is None:
        try:
            from paddleocr import PaddleOCR
            # Initialize PaddleOCR detector instance with english language
            _detector_instance = PaddleOCR(
                lang="en",
                show_log=False
            )
        except Exception as e:
            try:
                from paddleocr import PaddleOCR
                _detector_instance = PaddleOCR(lang="en")
            except Exception as e2:
                raise RuntimeError(f"PaddleOCR initialization failure: {e2}") from e
    return _detector_instance

def sort_reading_order(boxes_and_polys):
    """
    Sort bounding boxes top-to-bottom, left-to-right to preserve original reading order.
    Grouping algorithm clusters boxes into horizontal text lines using vertical centroid distance.
    """
    if not boxes_and_polys:
        return []

    items = []
    for item in boxes_and_polys:
        xmin, ymin, xmax, ymax = item['box']
        h = max(1, ymax - ymin)
        cy = (ymin + ymax) / 2.0
        items.append({**item, 'h': h, 'cy': cy})

    # Sort initially by ymin
    items.sort(key=lambda x: x['box'][1])

    avg_h = float(np.mean([x['h'] for x in items])) if items else 20.0
    line_threshold = max(10.0, avg_h * 0.5)

    lines = []
    for item in items:
        placed = False
        for line in lines:
            line_cy = float(np.mean([x['cy'] for x in line]))
            if abs(item['cy'] - line_cy) < line_threshold:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    # Sort lines top-to-bottom, and items within lines left-to-right
    lines.sort(key=lambda line: float(np.mean([x['box'][1] for x in line])))

    sorted_items = []
    for line in lines:
        line.sort(key=lambda x: x['box'][0])
        sorted_items.extend(line)

    return sorted_items

def detect_text(image_path: str, debug: bool = False):
    """
    Detect text regions in an image using PaddleOCR (detection mode only).
    
    Returns a list of dicts:
    [
        {
            'crop': PIL.Image.Image,
            'box': [xmin, ymin, xmax, ymax],
            'poly': polygon_points,
            'index': int,
            'crop_path': str or None
        }, ...
    ]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: '{image_path}'")

    cv_img = cv2.imread(image_path)
    if cv_img is None:
        raise ValueError(f"Failed to read image at '{image_path}' using OpenCV.")

    pil_img = Image.open(image_path).convert("RGB")
    detector = get_detector()

    raw_polys = []
    try:
        # Run PaddleOCR in detection-only mode (det=True, rec=False)
        ocr_res = detector.ocr(image_path, det=True, rec=False)
        if ocr_res and isinstance(ocr_res, list):
            if len(ocr_res) > 0 and isinstance(ocr_res[0], list):
                if len(ocr_res[0]) > 0 and (isinstance(ocr_res[0][0], list) or isinstance(ocr_res[0][0], np.ndarray)):
                    if len(ocr_res) == 1 and isinstance(ocr_res[0], list):
                        raw_polys = ocr_res[0]
                    else:
                        raw_polys = ocr_res
                elif isinstance(ocr_res[0], (list, np.ndarray)):
                    raw_polys = ocr_res
    except Exception as e:
        if debug:
            print(f"[DEBUG Detector] Primary det=True OCR call encountered issue: {e}")
        raw_polys = []

    # Fallback parsing if predict interface is used by newer PaddleOCR/Paddlex APIs
    if not raw_polys:
        try:
            results = detector.predict(image_path)
            for res in results:
                if hasattr(res, 'dt_polys') and res.dt_polys is not None:
                    raw_polys.extend(res.dt_polys)
                elif isinstance(res, dict) and 'dt_polys' in res:
                    raw_polys.extend(res['dt_polys'])
        except Exception:
            pass

    if not raw_polys:
        return []

    extracted = []
    img_h, img_w = cv_img.shape[:2]

    for poly in raw_polys:
        pts = np.array(poly, dtype=np.int32)
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)

        xmin = max(0, int(np.min(pts[:, 0])))
        ymin = max(0, int(np.min(pts[:, 1])))
        xmax = min(img_w, int(np.max(pts[:, 0])))
        ymax = min(img_h, int(np.max(pts[:, 1])))

        # Ensure box is valid (non-zero area)
        if xmax > xmin + 2 and ymax > ymin + 2:
            extracted.append({
                'box': [xmin, ymin, xmax, ymax],
                'poly': pts
            })

    # Sort boxes into standard reading order (top-to-bottom, left-to-right)
    sorted_regions = sort_reading_order(extracted)

    crops_dir = "crops"
    output_dir = "output"
    if debug:
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    debug_img = cv_img.copy() if debug else None

    results = []
    for idx, item in enumerate(sorted_regions):
        xmin, ymin, xmax, ymax = item['box']
        # Add slight padding around box for cleaner text crop
        pad = 2
        p_xmin = max(0, xmin - pad)
        p_ymin = max(0, ymin - pad)
        p_xmax = min(img_w, xmax + pad)
        p_ymax = min(img_h, ymax + pad)

        crop = pil_img.crop((p_xmin, p_ymin, p_xmax, p_ymax))

        crop_path = None
        if debug:
            crop_path = os.path.join(crops_dir, f"crop_{idx:03d}.png")
            crop.save(crop_path)
            cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(debug_img, str(idx), (xmin, max(15, ymin - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"[DEBUG] Region {idx}: Box=[{xmin}, {ymin}, {xmax}, {ymax}], Crop={crop_path}")

        results.append({
            'crop': crop,
            'box': [xmin, ymin, xmax, ymax],
            'poly': item['poly'],
            'index': idx,
            'crop_path': crop_path
        })

    if debug and debug_img is not None:
        bbox_path = os.path.join(output_dir, "detected_boxes.png")
        cv2.imwrite(bbox_path, debug_img)
        print(f"[DEBUG] Saved bounding box visualization to {bbox_path}")

    return results
