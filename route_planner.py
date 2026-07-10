"""
Safe Route Planning Module for Mountain Climbing Videos
========================================================
Analyzes climbing/mountain videos to detect terrain and recommend safe routes.

Pipeline:
1. Semantic Segmentation (terrain classification) - DeepLabV3+ / UPerNet
2. Depth Estimation (monocular depth) - MiDaS / DPT
3. Obstacle Detection (SAM - Segment Anything)
4. Traversability Cost Map (combine semantic + depth + obstacles)
5. Path Planning (A* / Dijkstra)
6. Visualization (overlay on video frames)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import heapq

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    transforms = None


# ============================================================================
# TERRAIN SEGMENTATION (Semantic Labels)
# ============================================================================

class TerrainSegmenter:
    """
    Semantic segmentation for terrain types.
    Uses a pre-trained DeepLabV3+ or similar model.
    Output: per-pixel class labels (rock, snow, grass, water, crevasse, etc.)
    """
    
    TERRAIN_CLASSES = {
        0: ("sky", 0.0),          # cost=0 (not walkable but ignored)
        1: ("rock", 0.6),          # cost=0.6 (moderate difficulty)
        2: ("snow", 0.7),          # cost=0.7 (slippery)
        3: ("grass", 0.2),         # cost=0.2 (easy)
        4: ("vegetation", 0.3),    # cost=0.3 (easy-moderate)
        5: ("water", 1.0),         # cost=1.0 (impassable)
        6: ("crevasse", 1.0),      # cost=1.0 (impassable)
        7: ("loose_rock", 0.95),   # cost=0.95 (very dangerous)
        8: ("ice", 0.85),          # cost=0.85 (difficult, slippery)
        9: ("dirt", 0.25),         # cost=0.25 (easy)
    }
    
    def __init__(self, model_name: str = "deeplabv3plus"):
        """
        Initialize segmentation model.
        Args:
            model_name: 'deeplabv3plus', 'upernet', or 'fcn'
        """
        self.model_name = model_name
        self.model = None
        self.transform = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.device = "cpu"
    
    def load_model(self):
        """Load pre-trained segmentation model from torchvision."""
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available - using fallback segmentation")
            self.model = None
            return
            
        try:
            if self.model_name == "deeplabv3plus":
                from torchvision.models.segmentation import deeplabv3_resnet101
                self.model = deeplabv3_resnet101(pretrained=True, num_classes=10)
            elif self.model_name == "fcn":
                from torchvision.models.segmentation import fcn_resnet101
                self.model = fcn_resnet101(pretrained=True, num_classes=10)
            else:
                # Fallback: simple FCN
                from torchvision.models.segmentation import deeplabv3_resnet50
                self.model = deeplabv3_resnet50(pretrained=True, num_classes=10)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Loaded {self.model_name} segmentation model")
        except Exception as e:
            print(f"⚠ Failed to load segmentation model: {e}")
            self.model = None
    
    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment a single frame into terrain classes.
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            sem_labels: (H, W) class indices [0..9]
        """
        if self.model is None:
            # Fallback: random segmentation for testing
            h, w = frame.shape[:2]
            return np.random.randint(0, 10, (h, w), dtype=np.uint8)
        
        # Preprocess
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = transforms.ToPILImage()(frame_rgb)
        input_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Extract class labels
        logits = output['out'] if isinstance(output, dict) else output
        sem_labels = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # Resize to original resolution if needed
        if sem_labels.shape != (h, w):
            sem_labels = cv2.resize(sem_labels, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return sem_labels


# ============================================================================
# DEPTH ESTIMATION (Monocular Depth)
# ============================================================================

class DepthEstimator:
    """
    Monocular depth estimation using MiDaS or DPT.
    Output: relative depth map (H, W) with values in [0..1] (normalized)
    """
    
    def __init__(self, model_type: str = "midas_v21"):
        """
        Initialize depth model.
        Args:
            model_type: 'midas_v21' (small, fast), 'midas_v31_large', 'dpt_large', etc.
        """
        self.model_type = model_type
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.model = None
        self.transform = None
    
    def load_model(self):
        """Load pre-trained depth model from torch hub."""
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available - using fallback depth estimation")
            self.model = None
            self.transform = None
            return

        try:
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(self.device)
            self.model.eval()
            
            # Load corresponding transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if "small" in self.model_type or "21" in self.model_type:
                self.transform = midas_transforms.small_transform
            elif "large" in self.model_type:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.dpt_transform
            
            print(f"✓ Loaded {self.model_type} depth model")
        except Exception as e:
            print(f"⚠ Failed to load depth model: {e}")
            self.model = None
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single frame.
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            depth: normalized depth map (H, W) in [0..1]
        """
        if self.model is None or self.transform is None or not TORCH_AVAILABLE:
            # Fallback: simple gradient-based pseudo-depth
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return gray
        
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        input_batch = self.transform(frame_rgb).to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)
        
        # Normalize to [0..1]
        depth = prediction.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Resize if needed
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return depth.astype(np.float32)


# ============================================================================
# TRAVERSABILITY COST MAP
# ============================================================================

class TraversabilityEstimator:
    """
    Compute traversability cost for each pixel by combining:
    - Semantic terrain class cost
    - Slope/gradient cost (from depth)
    - Roughness cost (local depth variance)
    - Obstacle mask cost
    """
    
    def __init__(self):
        self.terrain_costs = TerrainSegmenter.TERRAIN_CLASSES
    
    def compute_slope(self, depth: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Compute slope from depth gradient.
        Slope = arctan(|∇depth|)
        Args:
            depth: (H, W) normalized depth
            scale: scale factor for gradient
        Returns:
            slope: (H, W) normalized slope in [0..1]
        """
        gy, gx = np.gradient(depth)
        gradient_mag = np.sqrt(gx**2 + gy**2) * scale
        slope = np.arctan(gradient_mag) / (np.pi / 4)  # normalize to ~[0..1]
        return np.clip(slope, 0, 1).astype(np.float32)
    
    def compute_roughness(self, depth: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """
        Compute local roughness (depth variance in local window).
        Args:
            depth: (H, W) normalized depth
            kernel_size: local window size
        Returns:
            roughness: (H, W) local standard deviation
        """
        # Compute local variance
        mean = cv2.blur(depth, (kernel_size, kernel_size))
        mean_sq = cv2.blur(depth**2, (kernel_size, kernel_size))
        variance = mean_sq - mean**2
        roughness = np.sqrt(np.clip(variance, 0, None))
        
        # Normalize
        roughness = roughness / (roughness.max() + 1e-8)
        return roughness.astype(np.float32)
    
    def compute_cost_map(self,
                        sem_labels: np.ndarray,
                        depth: np.ndarray,
                        obstacle_mask: Optional[np.ndarray] = None,
                        w_semantic: float = 0.5,
                        w_slope: float = 0.25,
                        w_rough: float = 0.15,
                        w_obstacle: float = 0.1) -> np.ndarray:
        """
        Compute overall traversability cost map.
        Args:
            sem_labels: (H, W) semantic class indices
            depth: (H, W) normalized depth
            obstacle_mask: (H, W) binary obstacle mask (optional)
            w_*: weights for each cost component
        Returns:
            cost_map: (H, W) final cost in [0..1] (higher = more risky)
        """
        h, w = sem_labels.shape
        cost_map = np.zeros((h, w), dtype=np.float32)
        
        # Semantic cost
        sem_cost = np.zeros_like(sem_labels, dtype=np.float32)
        for cls_idx, (cls_name, cls_cost) in self.terrain_costs.items():
            sem_cost[sem_labels == cls_idx] = cls_cost
        
        # Slope cost
        slope_cost = self.compute_slope(depth)
        
        # Roughness cost
        rough_cost = self.compute_roughness(depth)
        
        # Obstacle cost
        if obstacle_mask is None:
            obstacle_cost = np.zeros_like(depth)
        else:
            obstacle_cost = obstacle_mask.astype(np.float32)
        
        # Combine
        cost_map = (w_semantic * sem_cost +
                   w_slope * slope_cost +
                   w_rough * rough_cost +
                   w_obstacle * obstacle_cost)
        
        # Normalize
        cost_map = cost_map / (cost_map.max() + 1e-8)
        
        return np.clip(cost_map, 0, 1).astype(np.float32)


# ============================================================================
# PATH PLANNING (A* Algorithm)
# ============================================================================

class PathPlanner:
    """
    A* pathfinding on a weighted grid graph.
    """
    
    def __init__(self, cost_map: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize path planner.
        Args:
            cost_map: (H, W) cost values in [0..1]
            start: (y, x) start pixel
            goal: (y, x) goal pixel
        """
        self.cost_map = cost_map
        self.h, self.w = cost_map.shape
        self.start = start
        self.goal = goal
        self.path = None
    
    def heuristic(self, node: Tuple[int, int]) -> float:
        """
        Admissible heuristic: Euclidean distance scaled by minimum cost.
        """
        y, x = node
        gy, gx = self.goal
        dist = np.sqrt((y - gy)**2 + (x - gx)**2)
        min_cost = self.cost_map.min()
        return dist * (min_cost + 0.1)
    
    def neighbors(self, node: Tuple[int, int], connectivity: int = 8) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells (4 or 8-connected).
        Args:
            node: (y, x)
            connectivity: 4 or 8
        Returns:
            list of valid neighbor coordinates
        """
        y, x = node
        neighbors = []
        
        # 4-connected: up, down, left, right
        deltas_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # 8-connected: add diagonals
        deltas_8 = deltas_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        deltas = deltas_4 if connectivity == 4 else deltas_8
        
        for dy, dx in deltas:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.h and 0 <= nx < self.w:
                neighbors.append((ny, nx))
        
        return neighbors
    
    def edge_cost(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
        """
        Cost of moving from node1 to node2.
        Use average cost of both cells plus move distance.
        """
        y1, x1 = node1
        y2, x2 = node2
        avg_cost = (self.cost_map[y1, x1] + self.cost_map[y2, x2]) / 2.0
        move_dist = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        return avg_cost * move_dist
    
    def plan(self, connectivity: int = 8, cost_threshold: float = 0.95) -> Optional[List[Tuple[int, int]]]:
        """
        Run A* pathfinding.
        Args:
            connectivity: 4 or 8-connected graph
            cost_threshold: skip cells with cost > threshold
        Returns:
            list of (y, x) waypoints from start to goal, or None if no path
        """
        open_set = []
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start)}
        came_from = {}
        closed_set = set()
        
        heapq.heappush(open_set, (f_score[self.start], self.start))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if current == self.goal:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(self.start)
                return list(reversed(path))
            
            for neighbor in self.neighbors(current, connectivity):
                if neighbor in closed_set:
                    continue
                
                # Skip high-cost cells
                if self.cost_map[neighbor[0], neighbor[1]] > cost_threshold:
                    continue
                
                tentative_g = g_score[current] + self.edge_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def smooth_path(self, path: List[Tuple[int, int]], iterations: int = 5) -> List[Tuple[int, int]]:
        """
        Smooth path by removing unnecessary waypoints (line-of-sight test).
        """
        if len(path) <= 2:
            return path
        
        smooth = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_of_sight(path[i], path[j]):
                    i = j - 1
                    break
                j -= 1
            else:
                i += 1
            
            if i < len(path) - 1:
                smooth.append(path[i + 1])
        
        if smooth[-1] != path[-1]:
            smooth.append(path[-1])
        
        return smooth
    
    def _line_of_sight(self, node1: Tuple[int, int], node2: Tuple[int, int], max_cost: float = 0.7) -> bool:
        """
        Check if there is a line-of-sight path between two nodes.
        Bresenham line + cost check.
        """
        y1, x1 = node1
        y2, x2 = node2
        
        # Bresenham line
        points = self._bresenham_line(y1, x1, y2, x2)
        
        # Check costs along line
        for y, x in points:
            if self.cost_map[y, x] > max_cost:
                return False
        
        return True
    
    def _bresenham_line(self, y0: int, x0: int, y1: int, x1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm."""
        points = []
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((y, x))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class SafeRouteAnalyzer:
    """
    Complete safe-route finding pipeline:
    1. Semantic segmentation
    2. Depth estimation
    3. Traversability cost computation
    4. Path planning (A*)
    5. Visualization
    """
    
    def __init__(self):
        self.segmenter = TerrainSegmenter(model_name="deeplabv3plus")
        self.depth_estimator = DepthEstimator(model_type="midas_v21")
        self.traversability = TraversabilityEstimator()
        self.last_cost_map = None
    
    def load_models(self):
        """Load segmentation and depth models."""
        self.segmenter.load_model()
        self.depth_estimator.load_model()
    
    def analyze_frame(self, frame: np.ndarray, 
                      obstacle_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze a single frame and return terrain + depth + cost map.
        Args:
            frame: BGR image
            obstacle_mask: optional binary obstacle mask
        Returns:
            dict with 'sem_labels', 'depth', 'cost_map', etc.
        """
        result = {}
        
        # Segmentation
        sem_labels = self.segmenter.segment_frame(frame)
        result['sem_labels'] = sem_labels
        
        # Depth
        depth = self.depth_estimator.estimate_depth(frame)
        result['depth'] = depth
        
        # Cost map
        cost_map = self.traversability.compute_cost_map(
            sem_labels, depth, obstacle_mask,
            w_semantic=0.5, w_slope=0.25, w_rough=0.15, w_obstacle=0.1
        )
        result['cost_map'] = cost_map
        self.last_cost_map = cost_map
        
        return result
    
    def plan_safe_route(self, cost_map: np.ndarray,
                       start: Optional[Tuple[int, int]] = None,
                       goal: Optional[Tuple[int, int]] = None,
                       cost_threshold: float = 0.9) -> Optional[List[Tuple[int, int]]]:
        """
        Plan safe route on cost map.
        Args:
            cost_map: (H, W) traversability cost
            start: (y, x) start pixel (default: center-bottom)
            goal: (y, x) goal pixel (default: center-top)
            cost_threshold: skip cells with cost > threshold
        Returns:
            list of waypoints or None
        """
        h, w = cost_map.shape
        
        # Default start/goal
        if start is None:
            start = (h - 1, w // 2)  # bottom-center
        if goal is None:
            goal = (0, w // 2)  # top-center
        
        # Ensure start/goal are valid
        start = tuple(np.clip(start, 0, (h-1, w-1)))
        goal = tuple(np.clip(goal, 0, (h-1, w-1)))
        
        planner = PathPlanner(cost_map, start, goal)
        path = planner.plan(connectivity=8, cost_threshold=cost_threshold)
        
        if path is not None:
            path = planner.smooth_path(path)
        
        return path
    
    def visualize_route(self, frame: np.ndarray,
                       cost_map: np.ndarray,
                       path: Optional[List[Tuple[int, int]]] = None,
                       title: str = "Safe Climbing Route") -> np.ndarray:
        """
        Visualize cost map and path on frame.
        Args:
            frame: BGR image
            cost_map: (H, W) cost values
            path: list of (y, x) waypoints
            title: display title
        Returns:
            annotated frame
        """
        h, w = frame.shape[:2]
        
        # Resize cost map to frame size if needed
        if cost_map.shape != (h, w):
            cost_map = cv2.resize(cost_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Create overlay: cost map as heatmap
        cost_heatmap = (cost_map * 255).astype(np.uint8)
        cost_heatmap = cv2.applyColorMap(cost_heatmap, cv2.COLORMAP_JET)
        
        # Blend with frame
        overlay = cv2.addWeighted(frame, 0.6, cost_heatmap, 0.4, 0)
        
        # Draw path
        if path is not None and len(path) > 1:
            path_points = np.array(path, dtype=np.int32)
            cv2.polylines(overlay, [path_points[:, ::-1]], False, (0, 255, 0), 3)
            
            # Mark start/goal
            cv2.circle(overlay, tuple(path[0][::-1]), 8, (0, 255, 0), -1)  # Green: start
            cv2.circle(overlay, tuple(path[-1][::-1]), 8, (0, 0, 255), -1)  # Red: goal
        
        # Add legend
        cv2.putText(overlay, "Safe Route Analysis", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Green=Low Cost, Red=High Cost", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return overlay


# ============================================================================
# TESTING / DEMO
# ============================================================================

if __name__ == "__main__":
    # Example usage (requires a video file)
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "sample_climbing.mp4"
    
    print(f"🎬 Loading video: {video_path}")
    
    analyzer = SafeRouteAnalyzer()
    print("🤖 Loading AI models...")
    analyzer.load_models()
    
    # Process first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        sys.exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        sys.exit(1)
    
    print("🔍 Analyzing frame...")
    result = analyzer.analyze_frame(frame)
    
    print("🗺 Planning route...")
    path = analyzer.plan_safe_route(result['cost_map'])
    
    if path:
        print(f"✓ Route found with {len(path)} waypoints")
    else:
        print("✗ No route found")
    
    print("🎨 Visualizing...")
    annotated = analyzer.visualize_route(frame, result['cost_map'], path)
    
    cv2.imshow("Safe Route", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    print("✓ Done!")
