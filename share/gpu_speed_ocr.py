import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
import logging
import re
import shutil
import os
import time
from datetime import datetime
from ultralytics import YOLO
import supervision as sv
import easyocr
from skimage import exposure
from skimage.filters import threshold_local
import concurrent.futures
import threading
import queue
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
# Optionally set environment variables for better GPU memory allocation on Jetson Nano
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'


class SimpleKalmanFilter:
    def __init__(self, initial_estimate: float, process_variance: float = 0.1, measurement_variance: float = 1.0):
        self.estimate = initial_estimate
        self.error_covariance = 1.0
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def update(self, measurement: float) -> float:
        self.error_covariance += self.process_variance
        K = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimate = self.estimate + K * (measurement - self.estimate)
        self.error_covariance = (1 - K) * self.error_covariance
        return self.estimate

def convert_speed(speed_mps, unit='mph'):
    conversions = {'mph': 2.23694, 'kmh': 3.6, 'mps': 1.0}
    return speed_mps * conversions.get(unit, 1.0)

class SpeedEstimator:
    def __init__(self, fps, window_size=5):
        self.fps = fps
        self.window_size = window_size
        self.positions = defaultdict(lambda: deque(maxlen=int(fps * 3)))
        self.speed_filters = {}
        self.previous_positions = {}

    def calculate_speed(self, tracker_id, point):
        self.positions[tracker_id].append(point)
        pts = np.array(self.positions[tracker_id])
        if pts.shape[0] < 2:
            return None
        recent_speeds = []
        t = np.arange(pts.shape[0])
        slope_x, _ = np.polyfit(t[-self.window_size:], pts[-self.window_size:, 0], 1)
        slope_y, _ = np.polyfit(t[-self.window_size:], pts[-self.window_size:, 1], 1)
        speed_reg_mps = np.sqrt(slope_x**2 + slope_y**2) * self.fps
        recent_speeds.append(speed_reg_mps)
        if tracker_id in self.previous_positions:
            displacement = np.linalg.norm(pts[-1] - self.previous_positions[tracker_id])
            speed_instant_mps = displacement * self.fps
            recent_speeds.append(speed_instant_mps)
        self.previous_positions[tracker_id] = pts[-1].copy()
        measured_speed_mps = sum(recent_speeds) / len(recent_speeds)
        if tracker_id not in self.speed_filters:
            self.speed_filters[tracker_id] = SimpleKalmanFilter(measured_speed_mps, process_variance=0.03, measurement_variance=0.15)
        filtered_speed_mps = self.speed_filters[tracker_id].update(measured_speed_mps)
        return convert_speed(filtered_speed_mps, 'mph')

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# ------------------------ Plate Extraction & OCR Preprocessing ------------------------

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def assess_image_quality(image):
    if image is None or image.size == 0:
        return 0
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)
    quality_score = (laplacian_var * 0.5 + brightness * 0.25 + contrast * 0.25) / 100
    return min(quality_score, 1.0)

def enhance_image_quality(image, upscale_factor=2):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        kernel_size = max(3, min(image.shape[:2]) // 50)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        cl = cv2.GaussianBlur(cl, (kernel_size, kernel_size), 0)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 10, 10, 7, 21)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        final = cv2.resize(sharpened, (0, 0), fx=upscale_factor, fy=upscale_factor,
                           interpolation=cv2.INTER_LANCZOS4)
        return final
    except Exception as e:
        logging.error(f"Enhance image quality failed: {e}")
        return image

def extract_plate(image, box, margin=0.3, upscale_factor=2):  # Increased margin
    try:
        x1, y1, x2, y2 = box
        h, w = image.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        
        # Increase margin for better plate capture
        dx = int(margin * box_w)
        dy = int(margin * box_h)
        x1e = max(0, x1 - dx)
        y1e = max(0, y1 - dy)
        x2e = min(w, x2 + dx)
        y2e = min(h, y2 + dy)
        
        if x2e <= x1e or y2e <= y1e:
            logging.warning("Invalid box dimensions after margin")
            return None
            
        crop = image[y1e:y2e, x1e:x2e]
        if crop.size == 0:
            logging.warning("Empty crop region")
            return None
            
        # Enhanced preprocessing
        enhanced = enhance_image_quality(crop, upscale_factor)
        return enhanced
    except Exception as e:
        logging.error(f"Plate extraction error: {e}")
        return None

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is not None:
        angle = 0
        for rho, theta in lines[0]:
            angle = theta * 180 / np.pi
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = -90 + angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    return image

def keep_largest_component(binary_image):
    if len(binary_image.shape) > 2:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    largest_label = 1
    largest_size = stats[1, cv2.CC_STAT_AREA]
    for label in range(2, num_labels):
        size = stats[label, cv2.CC_STAT_AREA]
        if size > largest_size:
            largest_size = size
            largest_label = label
    largest_mask = np.zeros_like(binary_image)
    largest_mask[labels == largest_label] = 255
    return largest_mask

def find_center_region(image, padding_percent=20):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    pad_x = int(width * padding_percent / 100)
    pad_y = int(height * padding_percent / 100)
    left = max(center_x - pad_x, 0)
    right = min(center_x + pad_x, width)
    top = max(center_y - pad_y, 0)
    bottom = min(center_y + pad_y, height)
    return left, right, top, bottom

def extract_text_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    left, right, top, bottom = find_center_region(image)
    mask = np.zeros_like(thresh)
    mask[top:bottom, left:right] = 255
    center_text = cv2.bitwise_and(thresh, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(center_text, cv2.MORPH_OPEN, kernel)
    return cleaned

def adaptive_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

def detect_text_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def enhance_image_colors(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    a = cv2.equalizeHist(a)
    b = cv2.equalizeHist(b)
    lab_image = cv2.merge([l, a, b])
    enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_image

def remove_white_pixels(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    non_white_mask = cv2.bitwise_not(white_mask)
    filtered_image = cv2.bitwise_and(image, image, mask=non_white_mask)
    kernel = np.ones((3,3), np.uint8)
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)
    return filtered_image

def sharpen_image(image):
    sharpening_kernel = np.array([[0, -1,  0],
                                  [-1,  5, -1],
                                  [0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(image, -1, sharpening_kernel)

def analyze_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def adjust_gamma(image):
    brightness = analyze_image_quality(image)
    if brightness < 100:
        gamma = 0.8
    elif brightness > 150:
        gamma = 1.2
    else:
        gamma = 1.0
    return exposure.adjust_gamma(image, gamma)

def fill_black_holes(binary_image):
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated

def enhanced_preprocess_image(image):
    denoised = denoise_image(image)
    gamma_corrected = adjust_gamma(denoised)
    enhanced = enhance_image_colors(gamma_corrected)
    filtered = remove_white_pixels(enhanced)
    thresholded = adaptive_thresholding(filtered)
    sharpened = sharpen_image(thresholded)
    return sharpened

# ------------------------- OCR Functions -------------------------

def find_base_angle(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    else:
        angle = -angle
    return angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def detect_two_digit_numbers(image, reader, pattern):
    results = reader.readtext(image, detail=1, paragraph=False, decoder='beamsearch')
    annotated_image = image.copy()
    valid_texts = []
    for (bbox, text, prob) in results:
        text_clean = text.strip().replace(" ", "")
        if pattern.match(text_clean):
            valid_texts.append(text_clean)
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(annotated_image, text_clean, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
    return valid_texts, annotated_image

# -------------------------- Global OCR Association Data --------------------------

forklift_plate = {}         # track_id -> OCR plate number
forklift_plate_count = {}   # track_id -> consecutive count
forklift_plate_finalized = set()  # track IDs for which OCR is finalized
lock = threading.Lock()

# Add these constants near other global variables
REQUIRED_CONSECUTIVE_MATCHES = 3  # Number of consecutive matches needed to finalize a plate
VALID_PLATE_RANGE = range(1, 100)  # Valid plate numbers from 01 to 99

def is_valid_plate_number(text):
    """Validate if text is a proper two-digit plate number."""
    try:
        num = int(text.strip())
        return num in VALID_PLATE_RANGE
    except ValueError:
        return False

# ------------------------- OCR Background Job Function -------------------------

def process_plate_job(frame, box, transformed_points, tracked_ids, pattern, reader):
    """Process a single plate detection
    
    Args:
        frame: The video frame
        box: Bounding box coordinates
        transformed_points: Transformed forklift points
        tracked_ids: List of tracked forklift IDs
        pattern: Regex pattern for plate number validation
        reader: EasyOCR reader instance
    """
    try:
        if not tracked_ids:
            return

        box_int = list(map(int, box))
        plate_img = extract_plate(frame, box_int, margin=0.3, upscale_factor=2)
        
        if plate_img is None:
            return
            
        best_text = None
        highest_confidence = 0
        
        for preprocess in [enhanced_preprocess_image, cv2.cvtColor]:
            try:
                processed_img = preprocess(plate_img) if preprocess != cv2.cvtColor else preprocess(plate_img, cv2.COLOR_BGR2GRAY)
                for angle in [0, -90, 90, 180]:
                    rotated = rotate_image(processed_img, angle)
                    results = reader.readtext(rotated, 
                                           detail=1,
                                           paragraph=False,
                                           decoder='beamsearch',
                                           batch_size=8)
                    
                    for (_, text, confidence) in results:
                        text_clean = text.strip().zfill(2)  # Ensure 2 digits with leading zero
                        if is_valid_plate_number(text_clean) and confidence > highest_confidence:
                            best_text = text_clean
                            highest_confidence = confidence
                            
            except Exception as e:
                logging.debug(f"Preprocessing method failed: {e}")
                continue
        
        if best_text and highest_confidence > 0.5:  # Added confidence threshold
            plate_point = np.array([(box_int[0] + box_int[2]) / 2, box_int[3]])
            
            if len(transformed_points) > 0:
                distances = np.linalg.norm(transformed_points - plate_point, axis=1)
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] < 20.0:
                    tracker_id = tracked_ids[nearest_idx]
                    
                    with lock:
                        if tracker_id in forklift_plate and forklift_plate[tracker_id] == best_text:
                            forklift_plate_count[tracker_id] = forklift_plate_count.get(tracker_id, 0) + 1
                            if forklift_plate_count[tracker_id] >= REQUIRED_CONSECUTIVE_MATCHES:
                                forklift_plate_finalized.add(tracker_id)
                                logging.info(f"Finalized plate {best_text} for forklift #{tracker_id}")
                        else:
                            # Reset counter if different number detected
                            forklift_plate[tracker_id] = best_text
                            forklift_plate_count[tracker_id] = 1
                            
    except Exception as e:
        logging.error(f"OCR job error: {e}")

# -------------------------- Optimized Main Pipeline --------------------------

plate_frame_queue = queue.Queue(maxsize=100)
PLATE_SAVE_DIR = Path("plate_frames")
PLATE_SAVE_DIR.mkdir(exist_ok=True)
ocr_processing_active = True

def process_plate_queue(reader, pattern):
    while ocr_processing_active:
        try:
            plate_data = plate_frame_queue.get(timeout=1.0)
            if plate_data is None:
                continue

            frame, box, frame_id, tracked_ids, transformed_points = plate_data
            process_plate_job(frame, box, transformed_points, tracked_ids, pattern, reader)
            plate_frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in plate queue processing: {e}")

def start_ocr_workers(num_workers=2):
    reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='./models',
                         download_enabled=True, recog_network='english_g2')
    pattern = re.compile(r'^[0-9]{1,2}$')
    
    workers = []
    for _ in range(num_workers):
        worker = threading.Thread(target=process_plate_queue, args=(reader, pattern))
        worker.daemon = True
        worker.start()
        workers.append(worker)
    return workers

# Update these constants for better performance
REQUIRED_CONSECUTIVE_MATCHES = 2  # Reduced from 3 for faster plate confirmation
OCR_WORKERS = max(4, os.cpu_count() - 1)  # Use more CPU cores
OCR_CONFIDENCE_THRESHOLD = 0.4  # Reduced threshold for more plate detections
OCR_SKIP_FRAMES = 3  # Process plates more frequently

class PlateManager:
    def __init__(self):
        self._plates = {}
        self._counts = {}
        self._finalized = set()
        self._lock = threading.Lock()

    def update_plate(self, tracker_id, plate_number, confidence):
        with self._lock:
            if tracker_id not in self._finalized:
                if tracker_id in self._plates and self._plates[tracker_id] == plate_number:
                    self._counts[tracker_id] = self._counts.get(tracker_id, 0) + 1
                    if self._counts[tracker_id] >= REQUIRED_CONSECUTIVE_MATCHES:
                        self._finalized.add(tracker_id)
                        logging.info(f"Finalized plate {plate_number} for forklift #{tracker_id}")
                        return True
                else:
                    self._plates[tracker_id] = plate_number
                    self._counts[tracker_id] = 1
            return False

    def get_plate(self, tracker_id):
        with self._lock:
            if tracker_id in self._plates:
                status = "✓" if tracker_id in self._finalized else "?"
                return f"{self._plates[tracker_id]}({status})"
            return None

plate_manager = PlateManager()

def process_plate_job(frame, box, transformed_points, tracked_ids, pattern, reader):
    try:
        if not tracked_ids:
            return

        box_int = list(map(int, box))
        plate_img = extract_plate(frame, box_int, margin=0.3, upscale_factor=2)
        
        if plate_img is None:
            return
            
        # Process image in parallel for different angles
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for preprocess in [enhanced_preprocess_image, cv2.cvtColor]:
                for angle in [0, -90, 90, 180]:
                    futures.append(executor.submit(
                        process_plate_variant,
                        plate_img, preprocess, angle, reader
                    ))
            
            results = [f.result() for f in futures if f.result()]
            
        if not results:
            return
            
        best_text, highest_confidence = max(results, key=lambda x: x[1])
        
        if best_text and highest_confidence > OCR_CONFIDENCE_THRESHOLD:
            plate_point = np.array([(box_int[0] + box_int[2]) / 2, box_int[3]])
            distances = np.linalg.norm(transformed_points - plate_point, axis=1)
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < 30.0:  # Increased distance threshold
                tracker_id = tracked_ids[nearest_idx]
                plate_manager.update_plate(tracker_id, best_text, highest_confidence)
                
    except Exception as e:
        logging.error(f"OCR job error: {e}")

def process_plate_variant(plate_img, preprocess, angle, reader):
    try:
        processed_img = preprocess(plate_img) if preprocess != cv2.cvtColor else preprocess(plate_img, cv2.COLOR_BGR2GRAY)
        rotated = rotate_image(processed_img, angle)
        results = reader.readtext(rotated, detail=1, paragraph=False, decoder='beamsearch', batch_size=8)
        
        best_result = None
        highest_confidence = 0
        
        for (_, text, confidence) in results:
            text_clean = text.strip().zfill(2)
            if is_valid_plate_number(text_clean) and confidence > highest_confidence:
                best_result = (text_clean, confidence)
                highest_confidence = confidence
                
        return best_result
    except Exception:
        return None

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description="Forklift Speed and Plate OCR Association")
    parser.add_argument("source_video_path", nargs="?", default=r"test3plate.mp4", type=str)
    parser.add_argument("--confidence_threshold", default=0.3, type=float)
    parser.add_argument("--iou_threshold", default=0.7, type=float)
    parser.add_argument("--target_fps", default=6.0, type=float)
    parser.add_argument("--debug", action="store_true", help="Display debug information")
    args = parser.parse_args()
    try:    
        video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    except Exception as e:
        logging.error(f"Failed to load video: {e}")
        return

    model = YOLO("best2.pt")
    # Send YOLO model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("No GPU detected; using CPU")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold)
    
    # Initialize speed estimator with the target FPS for consistent speed calculations
    speed_estimator = SpeedEstimator(args.target_fps)
    
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 3, position=sv.Position.BOTTOM_CENTER)
    SOURCE = np.array([[568, 310], [1321, 330], [1600, 1002], [264, 976], [568, 310]])
    TARGET = np.array([[0, 0], [12.192, 0], [12.192, 12.192], [0, 12.192], [0, 0]])
    polygon_zone = sv.PolygonZone(polygon=SOURCE[:4])  # Use only first 4 points for polygon zone
    view_transformer = ViewTransformer(source=SOURCE[:4], target=TARGET[:4])  # Use only first 4 points for transform
    temp_folder = os.path.join(os.getcwd(), "temp_plates")
    os.makedirs(temp_folder, exist_ok=True)
    
    # Initialize EasyOCR outside the loop for better performance
    reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='./models',
                           download_enabled=True, recog_network='english_g2')
    
    # More permissive pattern for testing
    pattern = re.compile(r'^[0-9]{1,2}$')
    
    # Create thread pool for OCR processing
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    futures = []
    
    # Precise frame timing control
    ocr_skip = 5  # Process OCR more frequently (every 5th frame)
    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    
    # Calculate the number of frames to skip for target FPS
    speed_skip = max(1, int(video_info.fps / args.target_fps))
    frame_counter = 0
    process_time = time.time()
    target_frame_time = 1.0 / args.target_fps  # Time per frame at target FPS

    # Start OCR worker threads
    ocr_workers = start_ocr_workers(num_workers=OCR_WORKERS)
    frame_id = 0

    try:
        while True:
            try:
                frame = next(frame_generator)
            except StopIteration:
                break
            
            if frame is None:
                break
            
            frame_counter += 1
            
            # Only process and display frames at target FPS
            if frame_counter % speed_skip == 0:
                current_process_time = time.time()
                
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > args.confidence_threshold]
                
                # Process forklifts (class 0)
                forklift_dets = detections[detections.class_id == 0]
                if not args.debug:
                    forklift_dets = forklift_dets[polygon_zone.trigger(forklift_dets)]
                forklift_dets = forklift_dets.with_nms(threshold=args.iou_threshold)
                tracked_dets = byte_track.update_with_detections(detections=forklift_dets)
                
                # Speed calculation and display
                forklift_points = tracked_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                transformed_points = view_transformer.transform_points(points=forklift_points)
                
                display_frame = frame.copy()
                forklift_labels = []
                
                for tracker_id, point in zip(tracked_dets.tracker_id, transformed_points):
                    speed = speed_estimator.calculate_speed(tracker_id, point)
                    label = f"Forklift #{tracker_id}"
                    
                    plate_info = plate_manager.get_plate(tracker_id)
                    if plate_info:
                        label += f" Plate: {plate_info}"
                    if speed is not None:
                        label += f" Speed: {speed:.1f} mph"
                    
                    forklift_labels.append(label)
                
                # Process plates only for unfinalized forklifts
                if frame_counter % ocr_skip == 0:
                    plate_dets = detections[detections.class_id == 1]
                    plate_dets = plate_dets.with_nms(threshold=args.iou_threshold)
                    
                    current_tracked_ids = set(tracked_dets.tracker_id)
                    unfinalized_ids = current_tracked_ids - forklift_plate_finalized

                    if unfinalized_ids and not plate_frame_queue.full():
                        for box in plate_dets.xyxy:
                            try:
                                plate_frame_queue.put_nowait((
                                    frame.copy(),
                                    box,
                                    frame_id,
                                    list(unfinalized_ids),
                                    transformed_points
                                ))
                            except queue.Full:
                                break

                # Annotate and display
                display_frame = box_annotator.annotate(scene=display_frame, detections=tracked_dets)
                display_frame = label_annotator.annotate(scene=display_frame, detections=tracked_dets, labels=forklift_labels)
                display_frame = trace_annotator.annotate(display_frame, tracked_dets)
                
                if args.debug:
                    fps = 1.0 / (time.time() - process_time)
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Tracking", display_frame)
                process_time = time.time()
                
                # Maintain target FPS
                frame_processing_time = time.time() - current_process_time
                if frame_processing_time < target_frame_time:
                    time.sleep(target_frame_time - frame_processing_time)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        # Cleanup
        global ocr_processing_active
        ocr_processing_active = False
        plate_frame_queue.join()
        for _ in range(len(ocr_workers)):
            plate_frame_queue.put(None)
        for worker in ocr_workers:
            worker.join(timeout=1.0)
        
        # Free GPU memory if used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        shutil.rmtree(str(PLATE_SAVE_DIR), ignore_errors=True)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set the GPU device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logging.info(f"Default GPU set: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("Running on CPU")
    main()