import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

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
        
        # Linear regression speed estimate
        t = np.arange(pts.shape[0])
        slope_x, _ = np.polyfit(t[-self.window_size:], pts[-self.window_size:, 0], 1)
        slope_y, _ = np.polyfit(t[-self.window_size:], pts[-self.window_size:, 1], 1)
        speed_reg_mps = np.sqrt(slope_x**2 + slope_y**2) * self.fps
        recent_speeds.append(speed_reg_mps)

        # Instantaneous displacement speed
        if tracker_id in self.previous_positions:
            displacement = np.linalg.norm(pts[-1] - self.previous_positions[tracker_id])
            speed_instant_mps = displacement * self.fps
            recent_speeds.append(speed_instant_mps)
        self.previous_positions[tracker_id] = pts[-1].copy()

        # Calculate average speed
        measured_speed_mps = sum(recent_speeds) / len(recent_speeds)

        # Apply Kalman filtering
        if tracker_id not in self.speed_filters:
            self.speed_filters[tracker_id] = SimpleKalmanFilter(
                measured_speed_mps, 
                process_variance=0.03,
                measurement_variance=0.15
            )
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
    conversions = {
        'mph': 2.23694,
        'kmh': 3.6,
        'mps': 1.0
    }
    return speed_mps * conversions.get(unit, 1.0)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forklift Speed Estimation using YOLO and Supervision"
    )
    parser.add_argument(
        "source_video_path",
        nargs="?",
        default=r"C:\Users\Kartikey.Tiwari\Downloads\ForkLfit\New folder\test3plate.mp4",
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "target_video_path",
        nargs="?",
        default=r"C:\Users\Kartikey.Tiwari\Downloads\ForkLfit\test_output1.mp4",
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--target_fps",
        default=6.0,
        help="Target processing FPS (for skipping frames)",
        type=float,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Display debug information and skip polygon filtering"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    
    # Initialize components
    model = YOLO("best2.pt")
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, 
        track_activation_threshold=args.confidence_threshold
    )
    speed_estimator = SpeedEstimator(args.target_fps)
    
    # Setup visualization
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 3,
        position=sv.Position.BOTTOM_CENTER,
    )

    # Setup transformation
    SOURCE = np.array([[568, 310], [1321, 330], [1600, 1002], [264, 976]])
    TARGET = np.array([[0, 0], [12.192, 0], [12.192, 12.192], [0, 12.192]])
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Process frames
    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    skip_frames = max(int(video_info.fps / args.target_fps), 1)
    frame_counter = 0

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            frame_counter += 1
            if frame_counter % skip_frames != 0:
                sink.write_frame(frame)
                continue

            # Detect and track forklifts
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[detections.class_id == 0]  # Only forklift class
            
            if not args.debug:
                detections = detections[polygon_zone.trigger(detections)]
            
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            # Calculate speeds
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_points = view_transformer.transform_points(points=points)
            
            labels = []
            for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, transformed_points)):
                speed = speed_estimator.calculate_speed(tracker_id, point)
                label = f"Forklift #{tracker_id}"
                if speed is not None:
                    label += f" Speed: {speed:.1f} mph"
                labels.append(label)

            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            if args.debug:
                # Draw debug information
                pts = SOURCE.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
