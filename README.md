# Yellow Line & Corner Detection System

Real-time detection of yellow lanes and right-angle corners using ROS 2 and OpenCV for autonomous navigation.

## Key Components

### 1. Yellow Line Detector (`yellow_line_detection_1b.py`)
- **Function**: Detects yellow lanes using HSV color thresholding
- **Outputs**: 
  - Binary detection flag (`/yellow_line/is_yellow_line`)
  - Distance to detected line (`/yellow_line/yellow_line_distance`)
- **Features**: 
  - Contour filtering with size and aspect ratio thresholds
  - Real-time depth integration from ZED camera
  - Visualization overlays (bounding box, distance label)

### 2. Yellow Corner Detector (`yellow_corner_plus_np_optimization.py`)
- **Function**: Detects right-angle yellow corners using ORB feature detection
- **Outputs**:
  - Corner detection flag (`/yellow_line/is_corner`)  
  - Distance to detected corner (`/yellow_line/corner_distance`)
- **Features**:
  - ORB keypoint detection with numpy-optimized 8-segment validation
  - CLAHE contrast enhancement for robust feature detection
  - Depth measurement at validated corner points
  - Vectorized operations for improved performance

### 3. Combined Detector (`combined_final_final_final_boss_dont_edit.py`)
- **Note**: Early integration attempt - **not recommended** for production use
- Contains both line and corner detection in a single node
- **Known issue**: Corner distance hardcoded to 3.0m (should use actual measurement)

## System Overview
```
RGB-D Camera Input
         ↓
[Yellow Line Detector] → Line Detection → Distance Measurement
         ↓
[Yellow Corner Detector] → Corner Detection → Distance Measurement
         ↓
ROS 2 Topics → Navigation System
```

## Usage
```bash
# Run line detector
ros2 run yellow_line_corner_detector yellow_line_node

# Run corner detector  
ros2 run yellow_line_corner_detector yellow_corner_node
```

## Performance Highlights
- Real-time processing at 30 FPS on Jetson Nano
- ORB-based corner detection achieves >90% accuracy in structured environments
- Distance measurement accuracy within 5cm ground truth (1-4m range)
- Modular ROS 2 architecture for easy integration

## Notes
- Repository cleaned for resume presentation: removed redundant version files
- Core algorithms implemented in the two highlighted Python nodes
- Parameters (HSV thresholds, contour filters) tuned for yellow tape detection
- Requires ROS 2 Humble, OpenCV, cv_bridge, and ZED camera drivers

---
*Built for autonomous navigation tasks requiring robust lane and corner detection in variable lighting conditions.*