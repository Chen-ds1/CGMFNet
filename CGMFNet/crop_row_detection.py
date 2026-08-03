


import colorsys
import csv
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import archs
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch
import yaml
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import KDTree, distance
from sklearn.cluster import DBSCAN


try:
    from evaluate import preprocess_pair as _orig_preprocess_pair
    _HAS_EVALUATE_PREPROCESS = True
except ImportError:
    _HAS_EVALUATE_PREPROCESS = False


def generate_distinct_colors(n, saturation=0.85, lightness=0.55):

    GOLDEN_RATIO = (np.sqrt(5) - 1) / 2
    hue = 0.0
    colors_rgb = []
    for _ in range(n):
        hue = (hue + GOLDEN_RATIO) % 1.0
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors_rgb.append((r, g, b))
    return colors_rgb


def estimate_normals(points, k=15):

    if len(points) < 4:
        reference = np.asarray([0, 0, 1], dtype=np.float32)
        normals = np.tile(reference, (len(points), 1))
        return normals, reference
    cov_global = np.cov(points.T)
    _, eigvecs = np.linalg.eigh(cov_global)
    ref_dir = eigvecs[:, -1]
    if ref_dir[2] < 0:
        ref_dir = -ref_dir
    ref_dir = ref_dir / np.linalg.norm(ref_dir)
    tree = KDTree(points)
    normals = np.zeros_like(points, dtype=np.float32)
    for i, pt in enumerate(points):
        _, idx = tree.query(pt, k=min(k + 1, len(points)))
        if len(idx) < 4:
            normals[i] = ref_dir
            continue
        neighbors = points[idx[1:]]
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid
        cov = centered.T @ centered / len(neighbors)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]
        if np.dot(normal, ref_dir) < 0:
            normal = -normal
        normals[i] = normal
    return normals, ref_dir


def compute_conditional_distance_matrix(points, normals,
                                        adhesion_dist=0.015,
                                        angle_thresh_deg=45.0,
                                        penalty_factor=5.0):


    N = len(points)
    D_eucl = distance.cdist(points, points, metric='euclidean')
    dot = np.abs(np.dot(normals, normals.T))
    dot = np.clip(dot, 0.0, 1.0)
    theta = np.arccos(dot)
    angle_thresh_rad = np.deg2rad(angle_thresh_deg)
    close_mask = D_eucl < adhesion_dist
    angle_diff_mask = theta > angle_thresh_rad
    penalty_mask = close_mask & angle_diff_mask
    D_hybrid = D_eucl.copy()
    D_hybrid[penalty_mask] *= penalty_factor
    np.fill_diagonal(D_hybrid, 0.0)
    return D_hybrid


def _fallback_preprocess_pair(rgb_img: np.ndarray,
                              depth_img: np.ndarray,
                              input_h: int,
                              input_w: int,
                              device: torch.device):
    rgb_img = cv2.resize(rgb_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    depth_img = cv2.resize(
        depth_img,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb_img = rgb_img.astype(np.float32) / 255.0
    depth_img = depth_img.astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    depth_tensor = (
        torch.from_numpy(depth_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    )
    return rgb_tensor, depth_tensor


def preprocess_pair(rgb_img: np.ndarray,
                    depth_img: np.ndarray,
                    input_h: int,
                    input_w: int,
                    device: torch.device):

    if _HAS_EVALUATE_PREPROCESS:
        return _orig_preprocess_pair(rgb_img, depth_img, input_h, input_w, device)
    else:
        return _fallback_preprocess_pair(rgb_img, depth_img, input_h, input_w, device)


class BaseProcessor:


    def __init__(self, model_name: str, threshold: Optional[float] = None):
        config_path = Path(f"models/{model_name}/config.yml")
        model_path  = Path(f"models/{model_name}/model.pth")
        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Model files not found: {config_path} or {model_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.model_config = yaml.safe_load(f)

        self.threshold = (
            threshold if threshold is not None
            else float(self.model_config.get("iou_threshold", 0.5))
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arch_name = self.model_config["arch"]
        self.model = archs.__dict__[arch_name](
            self.model_config["num_classes"],
            self.model_config["input_channels"],
            self.model_config["deep_supervision"],
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(str(model_path), map_location=self.device)
        )
        self.model.eval()

        self.input_h = self.model_config["input_h"]
        self.input_w = self.model_config["input_w"]

        self.align      = None
        self.temporal   = None
        self.color_map  = None


    @staticmethod
    def extract_pointcloud(depth_frame, color_frame,
                           mask: Optional[np.ndarray] = None):

        depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        fx, fy = depth_intrin.fx, depth_intrin.fy
        ppx, ppy = depth_intrin.ppx, depth_intrin.ppy

        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        h, w = depth_img.shape[:2]

        if mask is None:
            ys, xs = np.where(depth_img > 0)
        else:
            mask = np.asanyarray(mask)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        depths = depth_img[ys, xs].astype(np.float32)
        valid  = depths > 0
        xs, ys, depths = xs[valid], ys[valid], depths[valid] / 1000.0

        if len(xs) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        z = depths
        x = (xs - ppx) * z / fx
        y = (ys - ppy) * z / fy
        points_xyz = np.stack((x, y, z), axis=-1)
        colors_bgr = color_img[ys, xs]
        return points_xyz, colors_bgr


    def infer_mask(self, color_image_rgb: np.ndarray,
                   depth_colored_bgr: np.ndarray) -> np.ndarray:

        h, w = color_image_rgb.shape[:2]
        depth_colored_rgb = cv2.cvtColor(depth_colored_bgr, cv2.COLOR_BGR2RGB)

        rgb_tensor, depth_tensor = preprocess_pair(
            color_image_rgb, depth_colored_rgb,
            self.input_h, self.input_w, self.device,
        )

        with torch.inference_mode():
            output = self.model(rgb_tensor, depth_tensor)
            if isinstance(output, (list, tuple)):
                output = output[-1]
            probability = torch.sigmoid(output)[0, 0].cpu().numpy()

        probability = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (probability > self.threshold).astype(np.uint8) * 255
        return binary


    def process_frame_common(self, depth_frame, color_frame):


        if self.temporal is None:
            self.temporal = rs.temporal_filter()
        depth_filtered = self.temporal.process(depth_frame)


        if self.color_map is None:
            self.color_map = rs.colorizer()
        depth_colored_frame = self.color_map.process(depth_filtered)
        depth_colored_bgr   = np.asanyarray(depth_colored_frame.get_data())


        color_image = np.asanyarray(color_frame.get_data())
        color_format = color_frame.profile.format()
        if color_format == rs.format.bgr8:
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        else:
            color_image_rgb = color_image.copy()


        binary_mask = self.infer_mask(color_image_rgb, depth_colored_bgr)


        all_points, all_colors = self.extract_pointcloud(
            depth_filtered, color_frame, mask=binary_mask
        )


        if len(all_points) > 0:
            colors_rgb = all_colors[:, [2, 1, 0]] / 255.0
            green_mask = (colors_rgb[:, 1] > 0.1) & (colors_rgb[:, 0] < 0.4) & (colors_rgb[:, 2] < 0.4)
            green_points = all_points[green_mask]
        else:
            green_points = np.empty((0, 3), dtype=np.float32)

        return (True,
                depth_colored_bgr,
                color_image_rgb,
                green_points,
                binary_mask,
                depth_filtered)


class BagFileProcessor(BaseProcessor):


    def __init__(self, bag_path: str, model_name: str,
                 threshold: Optional[float] = None):
        super().__init__(model_name, threshold)

        if not Path(bag_path).exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_path, repeat_playback=False)
        config.enable_stream(rs.stream.depth)
        config.enable_stream(rs.stream.color)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.profile = self.pipeline.start(config)

    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            return (False,) + (None,) * 5

        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return (False,) + (None,) * 5

        return self.process_frame_common(depth_frame, color_frame)

    def release(self):
        self.pipeline.stop()


class CameraProcessor(BaseProcessor):


    def __init__(self, model_name: str, camera_resolution=(640, 480),
                 threshold: Optional[float] = None):
        super().__init__(model_name, threshold)

        self.pipeline = rs.pipeline()
        config = rs.config()
        width, height = camera_resolution
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.profile = self.pipeline.start(config)

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return (False,) + (None,) * 5

        return self.process_frame_common(depth_frame, color_frame)

    def release(self):
        self.pipeline.stop()


def cluster_and_extract_roots(points, params):


    if points is None or len(points) == 0:
        return []


    if params['voxel_size'] > 0:
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(points)
        pcd_down = pcd_tmp.voxel_down_sample(params['voxel_size'])
        pts = np.asarray(pcd_down.points)
    else:
        pts = points.copy()

    if len(pts) < 4:
        return []

    normals, _ = estimate_normals(pts, k=params['k_normal'])

    D = compute_conditional_distance_matrix(
        pts, normals,
        adhesion_dist=params['adhesion_dist'],
        angle_thresh_deg=params['angle_thresh'],
        penalty_factor=params['penalty_factor']
    )

    clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples'],
                        metric='precomputed').fit(D)
    labels = clustering.labels_
    unique_plants = set(labels) - {-1}
    root_positions = []

    for plant_id in unique_plants:
        pts_idx = np.where(labels == plant_id)[0]
        plant_pts = pts[pts_idx]
        if len(plant_pts) < params['min_points']:
            continue
        z_vals = plant_pts[:, 2]
        n_top = max(3, int(len(plant_pts) * 0.1))
        top_indices = np.argpartition(z_vals, -n_top)[-n_top:]
        root_point = np.mean(plant_pts[top_indices], axis=0)
        root_positions.append(root_point)

    return root_positions


def classify_rows_by_ipm(root_points, gap_threshold=0.12, fixed_ground_normal=None):

    N = len(root_points)
    if N < 2:
        return np.zeros(N, dtype=int), None, None, None

    centroid = np.mean(root_points, axis=0)
    centered = root_points - centroid

    if fixed_ground_normal is not None:
        ground_normal = np.array(fixed_ground_normal, dtype=np.float64)
        ground_normal /= np.linalg.norm(ground_normal)
    else:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        ground_normal = Vt[-1]
        if ground_normal[1] > 0:
            ground_normal = -ground_normal
        if abs(ground_normal[1]) < 0.2:
            ground_normal = np.array([0.0, -1.0, 0.0])

    camera_forward = np.array([0.0, 0.0, 1.0])
    lateral_axis = np.cross(camera_forward, ground_normal)
    lat_norm = np.linalg.norm(lateral_axis)
    lateral_axis = lateral_axis / lat_norm if lat_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    longitudinal = np.cross(ground_normal, lateral_axis)
    if np.dot(longitudinal, camera_forward) < 0:
        longitudinal = -longitudinal
    longitudinal /= np.linalg.norm(longitudinal)

    lateral_coords = np.dot(centered, lateral_axis)
    sorted_idx = np.argsort(lateral_coords)
    sorted_lat = lateral_coords[sorted_idx]
    gaps = np.diff(sorted_lat)
    split_pos = np.where(gaps > gap_threshold)[0]

    row_labels = np.zeros(N, dtype=int)
    start = 0
    row_id = 0
    for pos in split_pos:
        row_labels[sorted_idx[start:pos+1]] = row_id
        row_id += 1
        start = pos + 1
    row_labels[sorted_idx[start:]] = row_id
    return row_labels, longitudinal, lateral_axis, centroid


class ProcessingThread(QThread):
    new_frame = pyqtSignal(dict)
    log_message = pyqtSignal(str)

    def __init__(self, source_type, bag_path, params, model_name, camera_res=(640, 480)):
        super().__init__()
        self.source_type = source_type
        self.bag_path = bag_path
        self.params = params
        self.model_name = model_name
        self.camera_res = camera_res
        self.paused = False
        self.stopped = False
        self.processor = None

    def run(self):
        try:
            if self.source_type == 'bag':
                self.processor = BagFileProcessor(
                    self.bag_path, self.model_name, self.params.get('threshold', None)
                )
            else:
                self.processor = CameraProcessor(
                    self.model_name, self.camera_res, self.params.get('threshold', None)
                )

            intrinsic = None
            frame_count = 0
            while not self.stopped:
                while self.paused:
                    self.msleep(100)
                    if self.stopped:
                        return

                start_time = time.time()
                ret, depth, color, green_points, mask, depth_frame = self.processor.process_frame()
                if not ret:
                    break
                frame_count += 1
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                if intrinsic is None and depth_frame is not None:
                    intrinsic = depth_frame.profile.as_video_stream_profile().intrinsics

                root_points = []
                row_labels = []
                row_count = 0
                if green_points is not None and len(green_points) > 0:
                    root_points = cluster_and_extract_roots(green_points, self.params)
                    if len(root_points) >= 2 and intrinsic is not None:
                        pitch = self.params.get('camera_pitch', 40.0)
                        theta = math.radians(pitch)
                        fixed_n = [0.0, -math.cos(theta), -math.sin(theta)]
                        row_labels, _, _, _ = classify_rows_by_ipm(
                            np.array(root_points), self.params['ipm_gap'],
                            fixed_ground_normal=fixed_n
                        )
                        row_count = len(set(row_labels))

                data = {
                    'intrinsic': intrinsic,
                    'color_original': color,
                    'depth': depth,
                    'mask': mask,
                    'root_points': root_points,
                    'row_labels': row_labels,
                    'fps': fps,
                    'plant_count': len(root_points),
                    'row_count': row_count,
                    'process_time': elapsed
                }
                self.new_frame.emit(data)
                self.log_message.emit(
                    f"Frame {frame_count} | FPS: {fps:.1f} | Rows: {row_count} | Time: {elapsed*1000:.1f}ms"
                )

        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}")
        finally:
            if self.processor:
                self.processor.release()

    def stop(self):
        self.stopped = True

    def set_pause(self, pause):
        self.paused = pause


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crop Row Detection (CGMFNet)")
        self.setMinimumSize(1280, 800)


        self.params = {
            'eps': 0.04, 'min_samples': 10, 'min_points': 30,
            'k_normal': 5, 'voxel_size': 0.01, 'adhesion_dist': 0.015,
            'angle_thresh': 45.0, 'penalty_factor': 5.0, 'ipm_gap': 0.12,
            'camera_pitch': 60.0, 'threshold': None
        }
        self.bag_path = None
        self.thread = None
        self.current_data = None
        self.model_name = "CGMFNet-L4"
        self.source_mode = "bag"
        self.camera_width = 640
        self.camera_height = 480

        self.init_ui()


    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)


        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Bag File", "Live Camera"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        toolbar.addWidget(self.mode_combo)

        self.btn_open = QPushButton("Open Bag")
        self.btn_open.clicked.connect(self.open_bag)
        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.play)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_save = QPushButton("Save Frame")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_current_frame)

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_play)
        toolbar.addWidget(self.btn_pause)
        toolbar.addWidget(self.btn_stop)
        toolbar.addWidget(self.btn_save)
        main_layout.addLayout(toolbar)


        content = QHBoxLayout()
        self.img_label = QLabel("Waiting...")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(800, 600)
        self.img_label.setStyleSheet("background-color: black;")
        content.addWidget(self.img_label, stretch=3)


        right = QVBoxLayout()


        model_grp = QGroupBox("Model")
        ml = QHBoxLayout()
        ml.addWidget(QLabel("Name:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        if os.path.isdir("models"):
            try:
                available = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
                self.model_combo.addItems(available)
            except:
                pass
        self.model_combo.setCurrentText(self.model_name)
        ml.addWidget(self.model_combo)
        model_grp.setLayout(ml)
        right.addWidget(model_grp)


        param_grp = QGroupBox("Detection Parameters")
        grid = QGridLayout()
        row = 0
        grid.addWidget(QLabel("EPS:"), row, 0)
        self.eps_spin = QDoubleSpinBox(); self.eps_spin.setRange(0.01, 0.5); self.eps_spin.setSingleStep(0.01); self.eps_spin.setValue(self.params['eps'])
        grid.addWidget(self.eps_spin, row, 1); row += 1

        grid.addWidget(QLabel("Min Samples:"), row, 0)
        self.min_samp_spin = QSpinBox(); self.min_samp_spin.setRange(1, 50); self.min_samp_spin.setValue(self.params['min_samples'])
        grid.addWidget(self.min_samp_spin, row, 1); row += 1

        grid.addWidget(QLabel("Min Points:"), row, 0)
        self.min_pts_spin = QSpinBox(); self.min_pts_spin.setRange(10, 500); self.min_pts_spin.setValue(self.params['min_points'])
        grid.addWidget(self.min_pts_spin, row, 1); row += 1

        grid.addWidget(QLabel("Voxel Size:"), row, 0)
        self.voxel_spin = QDoubleSpinBox(); self.voxel_spin.setRange(0.001, 0.1); self.voxel_spin.setSingleStep(0.001); self.voxel_spin.setDecimals(3); self.voxel_spin.setValue(self.params['voxel_size'])
        grid.addWidget(self.voxel_spin, row, 1); row += 1

        grid.addWidget(QLabel("K Normal:"), row, 0)
        self.knormal_spin = QSpinBox(); self.knormal_spin.setRange(3, 30); self.knormal_spin.setValue(self.params['k_normal'])
        grid.addWidget(self.knormal_spin, row, 1); row += 1

        grid.addWidget(QLabel("Adhesion Dist:"), row, 0)
        self.adhesion_spin = QDoubleSpinBox(); self.adhesion_spin.setRange(0.005, 0.1); self.adhesion_spin.setSingleStep(0.001); self.adhesion_spin.setDecimals(3); self.adhesion_spin.setValue(self.params['adhesion_dist'])
        grid.addWidget(self.adhesion_spin, row, 1); row += 1

        grid.addWidget(QLabel("Angle Thresh:"), row, 0)
        self.angle_spin = QDoubleSpinBox(); self.angle_spin.setRange(10, 80); self.angle_spin.setSingleStep(1); self.angle_spin.setValue(self.params['angle_thresh'])
        grid.addWidget(self.angle_spin, row, 1); row += 1

        grid.addWidget(QLabel("Penalty Factor:"), row, 0)
        self.penalty_spin = QDoubleSpinBox(); self.penalty_spin.setRange(1.0, 20.0); self.penalty_spin.setSingleStep(0.5); self.penalty_spin.setValue(self.params['penalty_factor'])
        grid.addWidget(self.penalty_spin, row, 1); row += 1

        grid.addWidget(QLabel("IPM Gap:"), row, 0)
        self.ipmgap_spin = QDoubleSpinBox(); self.ipmgap_spin.setRange(0.05, 0.5); self.ipmgap_spin.setSingleStep(0.01); self.ipmgap_spin.setValue(self.params['ipm_gap'])
        grid.addWidget(self.ipmgap_spin, row, 1); row += 1

        grid.addWidget(QLabel("Camera Pitch (deg):"), row, 0)
        self.pitch_spin = QDoubleSpinBox(); self.pitch_spin.setRange(0.0, 90.0); self.pitch_spin.setSingleStep(1.0); self.pitch_spin.setValue(self.params['camera_pitch'])
        grid.addWidget(self.pitch_spin, row, 1); row += 1

        self.btn_apply = QPushButton("Apply Parameters")
        self.btn_apply.clicked.connect(self.apply_params)
        grid.addWidget(self.btn_apply, row, 0, 1, 2)
        param_grp.setLayout(grid)
        right.addWidget(param_grp)


        cam_grp = QGroupBox("Camera Resolution")
        cl = QHBoxLayout()
        cl.addWidget(QLabel("W:"))
        self.spin_cam_w = QSpinBox(); self.spin_cam_w.setRange(320, 1920); self.spin_cam_w.setSingleStep(10); self.spin_cam_w.setValue(self.camera_width)
        cl.addWidget(self.spin_cam_w)
        cl.addWidget(QLabel("H:"))
        self.spin_cam_h = QSpinBox(); self.spin_cam_h.setRange(240, 1080); self.spin_cam_h.setSingleStep(10); self.spin_cam_h.setValue(self.camera_height)
        cl.addWidget(self.spin_cam_h)
        cam_grp.setLayout(cl)
        right.addWidget(cam_grp)


        disp_grp = QGroupBox("Display")
        dl = QVBoxLayout()
        self.radio_color = QRadioButton("Color"); self.radio_color.setChecked(True)
        self.radio_depth = QRadioButton("Depth")
        self.radio_mask = QRadioButton("Mask")
        self.img_type_group = QButtonGroup()
        self.img_type_group.addButton(self.radio_color, 0)
        self.img_type_group.addButton(self.radio_depth, 1)
        self.img_type_group.addButton(self.radio_mask, 2)

        self.chk_show_lines = QCheckBox("Show Rows"); self.chk_show_lines.setChecked(True)
        self.chk_show_roots = QCheckBox("Show Points"); self.chk_show_roots.setChecked(True)

        def update_overlay_enabled(checked):
            enabled = self.radio_color.isChecked()
            self.chk_show_lines.setEnabled(enabled)
            self.chk_show_roots.setEnabled(enabled)
        self.radio_color.toggled.connect(update_overlay_enabled)
        update_overlay_enabled(True)

        dl.addWidget(self.radio_color)
        dl.addWidget(self.radio_depth)
        dl.addWidget(self.radio_mask)
        dl.addWidget(self.chk_show_lines)
        dl.addWidget(self.chk_show_roots)
        disp_grp.setLayout(dl)
        right.addWidget(disp_grp)

        content.addLayout(right, stretch=1)
        main_layout.addLayout(content)


        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        main_layout.addWidget(self.log_text)


        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_rows = QLabel("Rows: -")
        self.lbl_time = QLabel("Time: -")
        self.status_bar.addWidget(self.lbl_fps)
        self.status_bar.addWidget(self.lbl_rows)
        self.status_bar.addWidget(self.lbl_time)


        self.radio_color.toggled.connect(self.refresh_display)
        self.radio_depth.toggled.connect(self.refresh_display)
        self.radio_mask.toggled.connect(self.refresh_display)
        self.chk_show_lines.stateChanged.connect(self.refresh_display)
        self.chk_show_roots.stateChanged.connect(self.refresh_display)

        self.on_mode_changed(0)


    def on_mode_changed(self, index):
        self.source_mode = "bag" if index == 0 else "camera"
        self.btn_open.setVisible(index == 0)
        self.spin_cam_w.setEnabled(index == 1)
        self.spin_cam_h.setEnabled(index == 1)
        if self.source_mode == "camera":
            self.btn_play.setText("Start Camera")
            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.log("Switched to Live Camera mode.")
        else:
            self.btn_play.setText("Play")
            self.btn_play.setEnabled(self.bag_path is not None)
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_save.setEnabled(False)


    def draw_overlay_on_image(self, img_rgb):
        if self.current_data is None:
            return img_rgb

        intrinsic = self.current_data.get('intrinsic', None)
        roots = self.current_data.get('root_points', [])
        if intrinsic is None or len(roots) == 0:
            return img_rgb

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        show_roots = self.chk_show_roots.isEnabled() and self.chk_show_roots.isChecked()
        show_lines = self.chk_show_lines.isEnabled() and self.chk_show_lines.isChecked()

        row_labels = None
        color_map = None
        lines_2d = {}

        if (show_roots or show_lines) and len(roots) >= 2:
            pitch = self.params.get('camera_pitch', 40.0)
            theta = math.radians(pitch)
            fixed_n = [0.0, -math.cos(theta), -math.sin(theta)]
            root_arr = np.array(roots)
            row_labels, _, _, _ = classify_rows_by_ipm(
                root_arr, self.params['ipm_gap'], fixed_ground_normal=fixed_n
            )
            unique_rows = sorted(set(row_labels))
            if unique_rows:
                colors = generate_distinct_colors(len(unique_rows))
                color_map = {r: colors[i] for i, r in enumerate(unique_rows)}


        if show_roots and row_labels is not None and color_map is not None:
            for i, pt in enumerate(roots):
                px = rs.rs2_project_point_to_pixel(intrinsic, pt)
                r_id = row_labels[i]
                if r_id in color_map:
                    cr = color_map[r_id]
                    cb = (int(cr[2]*255), int(cr[1]*255), int(cr[0]*255))
                else:
                    cb = (0, 255, 255)
                cv2.circle(img_bgr, (int(px[0]), int(px[1])), 3, cb, -1)


        if show_lines and row_labels is not None and color_map is not None:
            row_pts_3d = defaultdict(list)
            for i, r in enumerate(roots):
                row_pts_3d[row_labels[i]].append(r)

            h, w = img_bgr.shape[:2]
            for r_id, pts_3d in row_pts_3d.items():
                if len(pts_3d) < 3:
                    continue
                pts_2d = []
                for p in pts_3d:
                    px, py = rs.rs2_project_point_to_pixel(intrinsic, p)
                    if np.isfinite(px) and np.isfinite(py):
                        pts_2d.append((px, py))
                pts_2d = np.array(pts_2d, dtype=np.float32).reshape(-1, 2)
                if len(pts_2d) < 2:
                    continue

                line = cv2.fitLine(pts_2d, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x0, y0 = line[0][0], line[1][0], line[2][0], line[3][0]


                t_vals = []
                if vx != 0:
                    for x_bound in [0, w-1]:
                        t = (x_bound - x0) / vx
                        y = y0 + t * vy
                        if 0 <= y <= h-1: t_vals.append(t)
                if vy != 0:
                    for y_bound in [0, h-1]:
                        t = (y_bound - y0) / vy
                        x = x0 + t * vx
                        if 0 <= x <= w-1: t_vals.append(t)
                if len(t_vals) < 2:
                    continue

                t_min, t_max = min(t_vals), max(t_vals)
                pt1 = (int(x0 + t_min*vx), int(y0 + t_min*vy))
                pt2 = (int(x0 + t_max*vx), int(y0 + t_max*vy))
                pt1 = (max(0, min(w-1, pt1[0])), max(0, min(h-1, pt1[1])))
                pt2 = (max(0, min(w-1, pt2[0])), max(0, min(h-1, pt2[1])))

                lines_2d[r_id] = {
                    'vx': float(vx), 'vy': float(vy),
                    'x0': float(x0), 'y0': float(y0),
                    'pt1': pt1, 'pt2': pt2
                }

                cr = color_map[r_id]
                cb = (int(cr[2]*255), int(cr[1]*255), int(cr[0]*255))
                cv2.line(img_bgr, pt1, pt2, cb, 1, cv2.LINE_AA)

        self.current_data['lines_2d'] = lines_2d
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


    def refresh_display(self):
        if self.current_data is None:
            return
        data = self.current_data

        if self.radio_color.isChecked():
            base = data['color_original'].copy()
            if self.chk_show_lines.isEnabled() or self.chk_show_roots.isEnabled():
                base = self.draw_overlay_on_image(base)
        elif self.radio_depth.isChecked():
            base = data['depth']
        else:
            base = data['mask']

        if base.dtype != np.uint8:
            base = np.clip(base, 0, 255).astype(np.uint8)

        if base.ndim == 3 and base.shape[2] == 3:
            h, w, ch = base.shape
            qimg = QImage(base.data, w, h, 3*w, QImage.Format_RGB888)
        else:
            h, w = base.shape[:2]
            if base.ndim == 2:
                base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
            else:
                base_rgb = base
            qimg = QImage(base_rgb.data, w, h, 3*w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.img_label.setPixmap(pixmap)

        self.lbl_fps.setText(f"FPS: {data['fps']:.1f}")
        self.lbl_rows.setText(f"Rows: {data['row_count']}")
        self.lbl_time.setText(f"Time: {data['process_time']*1000:.0f}ms")


    def log(self, msg):
        self.log_text.append(msg)


    def open_bag(self):
        if self.source_mode != "bag":
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Bag File", "", "Bag files (*.bag)",
            options=QFileDialog.DontUseNativeDialog
        )
        if path:
            self.bag_path = path
            self.log(f"Selected: {path}")
            self.btn_play.setEnabled(True)


    def play(self):
        if self.source_mode == "bag" and self.bag_path is None:
            QMessageBox.warning(self, "Info", "Please open a bag file first")
            return

        self.model_name = self.model_combo.currentText().strip()
        if not self.model_name:
            QMessageBox.warning(self, "Info", "Please enter a model name")
            return
        config_path = f"models/{self.model_name}/config.yml"
        model_path  = f"models/{self.model_name}/model.pth"
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", f"Model files not found:\n{config_path}\n{model_path}")
            return

        self.camera_width = self.spin_cam_w.value()
        self.camera_height = self.spin_cam_h.value()

        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()

        self.thread = ProcessingThread(
            self.source_mode, self.bag_path, self.params, self.model_name,
            camera_res=(self.camera_width, self.camera_height)
        )
        self.thread.new_frame.connect(self.update_frame)
        self.thread.log_message.connect(self.log)
        self.thread.start()

        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.log("Playback started...")


    def pause(self):
        if self.thread and self.thread.isRunning():
            self.thread.set_pause(True)
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.log("Paused")


    def stop(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.log("Stopped")
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)


    def apply_params(self):
        self.params['eps'] = self.eps_spin.value()
        self.params['min_samples'] = self.min_samp_spin.value()
        self.params['min_points'] = self.min_pts_spin.value()
        self.params['voxel_size'] = self.voxel_spin.value()
        self.params['k_normal'] = self.knormal_spin.value()
        self.params['adhesion_dist'] = self.adhesion_spin.value()
        self.params['angle_thresh'] = self.angle_spin.value()
        self.params['penalty_factor'] = self.penalty_spin.value()
        self.params['ipm_gap'] = self.ipmgap_spin.value()
        self.params['camera_pitch'] = self.pitch_spin.value()
        self.log("Parameters updated.")


    def update_frame(self, data):
        self.current_data = data
        self.refresh_display()


    def save_current_frame(self):
        if self.current_data is None:
            QMessageBox.warning(self, "Info", "No data to save")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Select Save Directory",
            options=QFileDialog.DontUseNativeDialog
        )
        if not directory:
            return

        base_name = time.strftime("%Y%m%d_%H%M%S")
        data = self.current_data


        if self.radio_color.isChecked():
            base = data['color_original'].copy()
            if self.chk_show_lines.isEnabled() or self.chk_show_roots.isEnabled():
                base = self.draw_overlay_on_image(base)
        elif self.radio_depth.isChecked():
            base = data['depth']
        else:
            base = data['mask']
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

        if base.dtype != np.uint8:
            base = np.clip(base, 0, 255).astype(np.uint8)

        if base.ndim == 3 and base.shape[2] == 3:
            img_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = base
        img_path = os.path.join(directory, f"frame_{base_name}.png")
        cv2.imwrite(img_path, img_bgr)
        self.log(f"Image saved: {img_path}")


        root_points = data.get('root_points', [])
        if root_points:
            csv_path = os.path.join(directory, f"roots_{base_name}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z"])
                writer.writerows(root_points)
            self.log(f"Root points saved: {csv_path}")


        lines_2d = data.get('lines_2d', {})
        if lines_2d:
            csv_path = os.path.join(directory, f"row_lines_{base_name}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["row_id", "vx", "vy", "x0", "y0", "pt1_x", "pt1_y", "pt2_x", "pt2_y"])
                for r_id, info in lines_2d.items():
                    pt1 = info.get('pt1', (None, None))
                    pt2 = info.get('pt2', (None, None))
                    writer.writerow([r_id, info['vx'], info['vy'], info['x0'], info['y0'],
                                     pt1[0], pt1[1], pt2[0], pt2[1]])
            self.log(f"Row line parameters saved: {csv_path}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

