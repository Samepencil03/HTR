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

def sort_reading_order(boxes_and_polys, merge_horizontal: bool = True):
    """
    Sort bounding boxes top-to-bottom, left-to-right to preserve original reading order.
    Clusters boxes into horizontal text lines using vertical centroid distance.
    Merges adjacent word bounding boxes in the same line to form complete text-line crops for TrOCR.
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

    # Sort lines top-to-bottom
    lines.sort(key=lambda line: float(np.mean([x['box'][1] for x in line])))

    merged_items = []
    for line in lines:
        line.sort(key=lambda x: x['box'][0])

        if not merge_horizontal or len(line) <= 1:
            merged_items.extend(line)
            continue

        line_avg_h = float(np.mean([x['h'] for x in line]))
        max_gap = max(15.0, line_avg_h * 1.5)

        curr = line[0]
        for next_item in line[1:]:
            c_xmin, c_ymin, c_xmax, c_ymax = curr['box']
            n_xmin, n_ymin, n_xmax, n_ymax = next_item['box']

            gap = n_xmin - c_xmax
            if gap <= max_gap:
                m_xmin = min(c_xmin, n_xmin)
                m_ymin = min(c_ymin, n_ymin)
                m_xmax = max(c_xmax, n_xmax)
                m_ymax = max(c_ymax, n_ymax)
                curr['box'] = [m_xmin, m_ymin, m_xmax, m_ymax]
                curr['h'] = m_ymax - m_ymin
                curr['cy'] = (m_ymin + m_ymax) / 2.0
            else:
                merged_items.append(curr)
                curr = next_item
        merged_items.append(curr)

    return merged_items

def detect_text(image_path: str, debug: bool = False, merge_lines: bool = True):
    """
    Detect text regions in an image using PaddleOCR (detection mode only).
    Merges word boxes into full text lines and applies dynamic padding for HTR line recognition.
    
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

        if xmax > xmin + 2 and ymax > ymin + 2:
            extracted.append({
                'box': [xmin, ymin, xmax, ymax],
                'poly': pts
            })

    # Sort and merge boxes into full text lines in reading order
    sorted_regions = sort_reading_order(extracted, merge_horizontal=merge_lines)

    crops_dir = "crops"
    output_dir = "output"
    if debug:
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    debug_img = cv_img.copy() if debug else None

    results = []
    for idx, item in enumerate(sorted_regions):
        xmin, ymin, xmax, ymax = item['box']
        box_h = ymax - ymin
        box_w = xmax - xmin
        
        # Dynamic padding to avoid clipping letter edges
        pad_h = max(4, int(0.08 * box_h))
        pad_w = max(4, int(0.08 * box_w))
        
        p_xmin = max(0, xmin - pad_w)
        p_ymin = max(0, ymin - pad_h)
        p_xmax = min(img_w, xmax + pad_w)
        p_ymax = min(img_h, ymax + pad_h)

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
