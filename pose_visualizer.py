"""
Pose Visualization Module
==========================
Visualize MoveNet keypoints and detect climbing-specific issues with visual feedback.
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    
import numpy as np
from typing import Tuple, List, Dict


class PoseVisualizer:
    """Visualize pose keypoints and detected issues on video frames."""
    
    # MoveNet keypoint names (17 joints)
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # Skeleton connections (joint pairs to draw lines)
    SKELETON_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Colors
    COLOR_GOOD = (0, 255, 0)  # Green
    COLOR_WARNING = (0, 165, 255)  # Orange
    COLOR_BAD = (0, 0, 255)  # Red
    COLOR_SKELETON = (255, 0, 0)  # Blue for skeleton
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for pose visualization. Please install it: pip install opencv-python")
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray,
                      threshold: float = 0.3) -> np.ndarray:
        """
        Draw pose keypoints on frame.
        Args:
            frame: BGR image
            keypoints: (17, 3) array [y, x, confidence]
            threshold: confidence threshold for drawing
        Returns:
            frame with keypoints drawn
        """
        h, w = frame.shape[:2]
        
        # Draw skeleton (lines between joints)
        for edge in self.SKELETON_EDGES:
            pt1_idx, pt2_idx = edge
            kpt1 = keypoints[pt1_idx]
            kpt2 = keypoints[pt2_idx]
            
            if kpt1[2] > threshold and kpt2[2] > threshold:
                y1, x1 = int(kpt1[0] * h), int(kpt1[1] * w)
                y2, x2 = int(kpt2[0] * h), int(kpt2[1] * w)
                cv2.line(frame, (x1, y1), (x2, y2), self.COLOR_SKELETON, 2)
        
        # Draw keypoint circles
        for i, kpt in enumerate(keypoints):
            if kpt[2] > threshold:
                y, x = int(kpt[0] * h), int(kpt[1] * w)
                color = self.COLOR_GOOD if kpt[2] > 0.5 else self.COLOR_WARNING
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 2)
                
                # Draw label
                cv2.putText(frame, self.KEYPOINT_NAMES[i], (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_issue_overlay(self, frame: np.ndarray, issues: List[str],
                          issue_severity: Dict[str, str] = None) -> np.ndarray:
        """
        Draw detected issues as overlay on frame.
        Args:
            frame: BGR image
            issues: list of issue names detected
            issue_severity: dict mapping issue to 'warning'/'critical'
        Returns:
            frame with issue overlay
        """
        if issue_severity is None:
            issue_severity = {}
        
        y_offset = 30
        for issue in issues:
            severity = issue_severity.get(issue, 'warning')
            color = self.COLOR_BAD if severity == 'critical' else self.COLOR_WARNING
            
            # Draw background rectangle
            text = f"⚠ {issue.replace('_', ' ').title()}"
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            cv2.rectangle(frame, (10, y_offset - 20), 
                         (20 + text_size[0], y_offset + text_size[1]), 
                         color, -1)
            cv2.putText(frame, text, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            y_offset += 35
        
        return frame
    
    def analyze_and_visualize(self, frame: np.ndarray, keypoints: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Analyze pose and return annotated frame with issues.
        Args:
            frame: BGR image
            keypoints: (17, 3) array
        Returns:
            (annotated_frame, issues_list)
        """
        issues = []
        issue_severity = {}
        
        h, w = frame.shape[:2]
        
        # Extract keypoints
        left_ankle, right_ankle = keypoints[15], keypoints[16]
        left_hip, right_hip = keypoints[11], keypoints[12]
        left_wrist, right_wrist = keypoints[9], keypoints[10]
        left_knee, right_knee = keypoints[13], keypoints[14]
        left_elbow, right_elbow = keypoints[7], keypoints[8]
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        nose = keypoints[0]
        
        # Check confidence
        if nose[2] < 0.3:
            return frame, ["person_not_detected"]
        
        # Issue 1: Wide Stance
        if abs(left_ankle[0] - right_ankle[0]) > 0.35 and left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
            issues.append("wide_stance")
            issue_severity["wide_stance"] = "warning"
            # Highlight ankles
            y1, x1 = int(left_ankle[0] * h), int(left_ankle[1] * w)
            y2, x2 = int(right_ankle[0] * h), int(right_ankle[1] * w)
            cv2.circle(frame, (x1, y1), 8, self.COLOR_BAD, 2)
            cv2.circle(frame, (x2, y2), 8, self.COLOR_BAD, 2)
        
        # Issue 2: Hips Away from Wall
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        if avg_hip_y > 0.65 and left_hip[2] > 0.3 and right_hip[2] > 0.3:
            issues.append("hips_away_from_wall")
            issue_severity["hips_away_from_wall"] = "critical"
            # Highlight hips
            y1, x1 = int(left_hip[0] * h), int(left_hip[1] * w)
            y2, x2 = int(right_hip[0] * h), int(right_hip[1] * w)
            cv2.circle(frame, (x1, y1), 8, self.COLOR_BAD, 2)
            cv2.circle(frame, (x2, y2), 8, self.COLOR_BAD, 2)
        
        # Issue 3: Poor Hand Usage
        if (left_wrist[2] < 0.25 or right_wrist[2] < 0.25):
            issues.append("poor_hand_usage")
            issue_severity["poor_hand_usage"] = "warning"
            if left_wrist[2] < 0.25:
                y, x = int(left_wrist[0] * h), int(left_wrist[1] * w)
                cv2.circle(frame, (x, y), 8, self.COLOR_BAD, 2)
            if right_wrist[2] < 0.25:
                y, x = int(right_wrist[0] * h), int(right_wrist[1] * w)
                cv2.circle(frame, (x, y), 8, self.COLOR_BAD, 2)
        
        # Issue 4: Unstable Knees
        if left_knee[2] > 0.3 and right_knee[2] > 0.3:
            knee_dist = abs(left_knee[0] - right_knee[0])
            hip_dist = abs(left_hip[0] - right_hip[0])
            if knee_dist > hip_dist * 1.5:
                issues.append("unstable_knees")
                issue_severity["unstable_knees"] = "warning"
                y1, x1 = int(left_knee[0] * h), int(left_knee[1] * w)
                y2, x2 = int(right_knee[0] * h), int(right_knee[1] * w)
                cv2.circle(frame, (x1, y1), 8, self.COLOR_BAD, 2)
                cv2.circle(frame, (x2, y2), 8, self.COLOR_BAD, 2)
        
        # Issue 5: Overreaching
        if (left_elbow[2] > 0.3 and right_elbow[2] > 0.3):
            left_arm_length = abs(left_shoulder[1] - left_elbow[1])
            right_arm_length = abs(right_shoulder[1] - right_elbow[1])
            if left_arm_length > 0.2 or right_arm_length > 0.2:
                issues.append("overreaching")
                issue_severity["overreaching"] = "warning"
                if left_arm_length > 0.2:
                    y, x = int(left_elbow[0] * h), int(left_elbow[1] * w)
                    cv2.circle(frame, (x, y), 8, self.COLOR_BAD, 2)
                if right_arm_length > 0.2:
                    y, x = int(right_elbow[0] * h), int(right_elbow[1] * w)
                    cv2.circle(frame, (x, y), 8, self.COLOR_BAD, 2)
        
        # Draw keypoints
        frame = self.draw_keypoints(frame, keypoints)
        
        # Draw issue overlay
        if issues:
            frame = self.draw_issue_overlay(frame, issues, issue_severity)
        
        return frame, issues


class ImprovementAdvisor:
    """Provide personalized climbing improvement advice based on detected issues."""
    
    IMPROVEMENT_TIPS = {
        "wide_stance": {
            "issue": "Wide Stance Detected",
            "description": "Your feet are too far apart, reducing stability and control",
            "why_bad": "Wide stance increases fatigue and reduces your ability to maintain balance on the wall",
            "key_points_to_avoid": [
                "❌ DON'T spread feet wider than shoulder width",
                "❌ DON'T place feet on the same plane (one above the other is better)",
                "❌ DON'T sacrifice precision for speed",
                "❌ DON'T climb with stiff legs - move dynamically",
                "❌ DON'T ignore small footholds - use them for precision",
            ],
            "improvement_steps": [
                "1️⃣ Keep feet shoulder-width apart or closer",
                "2️⃣ Practice placing feet on small footholds (precision footwork)",
                "3️⃣ Maintain a 'pigeon-toed' position for better wall contact",
                "4️⃣ Focus on quiet, controlled foot placements",
                "5️⃣ Climb on routes with small footholds to train this skill",
            ],
            "drill": "Silent Climbing Drill: Climb focusing on making no noise with your feet"
        },
        "hips_away_from_wall": {
            "issue": "Hips Away from Wall",
            "description": "Your hips are too far from the climbing surface",
            "why_bad": "This dramatically increases arm strain and fatigue - one of the biggest mistakes!",
            "key_points_to_avoid": [
                "❌ DON'T lean away from the wall with your hips",
                "❌ DON'T use only arm strength - engage your legs!",
                "❌ DON'T keep arms straight when climbing - bend elbows",
                "❌ DON'T reach far - move feet up instead",
                "❌ DON'T climb in a 'flag' position unnecessarily",
            ],
            "improvement_steps": [
                "1️⃣ Consciously push your hips closer to the wall",
                "2️⃣ Rotate your body laterally to get hips in",
                "3️⃣ Think 'hips first, then reach for holds'",
                "4️⃣ Practice hip flexibility exercises (lunges, pigeon pose)",
                "5️⃣ Climb with a spotter pointing out when your hips drop",
            ],
            "drill": "Wall Proximity Drill: Climb at 2-3 feet from wall, focus on maintaining contact"
        },
        "poor_hand_usage": {
            "issue": "Inconsistent Hand Usage",
            "description": "One or both hands are losing confidence in the hold",
            "why_bad": "Poor hand usage reduces your ability to push off holds and makes moves unpredictable",
            "key_points_to_avoid": [
                "❌ DON'T skip good handholds - use every available hold",
                "❌ DON'T rely on only one hand (imbalance)",
                "❌ DON'T grip too hard - use efficient hand pressure",
                "❌ DON'T lose focus between moves",
                "❌ DON'T move hands and feet simultaneously (lose stability)",
            ],
            "improvement_steps": [
                "1️⃣ Use all available handholds - don't skip holds",
                "2️⃣ Maintain three-point contact (2 hands + 1 foot or 1 hand + 2 feet)",
                "3️⃣ Practice transitions smoothly between holds",
                "4️⃣ Grip training: use hangboard 3x per week",
                "5️⃣ Climb routes with crimpy holds to improve hand strength",
            ],
            "drill": "Three-Point Contact Drill: Every move must have 3+ contact points"
        },
        "unstable_knees": {
            "issue": "Knee Instability",
            "description": "Your knees are spreading apart or collapsing inward",
            "why_bad": "Unstable knees indicate weak leg positioning and reduce efficiency",
            "key_points_to_avoid": [
                "❌ DON'T let knees cave inward (valgus collapse)",
                "❌ DON'T spread knees too wide apart",
                "❌ DON'T ignore knee pain - check your form",
                "❌ DON'T climb on weak legs - strengthen first",
                "❌ DON'T twist your body excessively",
            ],
            "improvement_steps": [
                "1️⃣ Keep knees aligned with your body line",
                "2️⃣ Avoid letting knees cave inward ('valgus collapse')",
                "3️⃣ Strengthen leg muscles: squats, lunges, calf raises",
                "4️⃣ Practice standing on one leg for balance",
                "5️⃣ Climb with deliberate knee positioning - don't rush",
            ],
            "drill": "One-Leg Balance Drill: Stand on each leg for 60 seconds before climbing"
        },
        "overreaching": {
            "issue": "Overreaching",
            "description": "Your arms are fully extended when reaching for holds",
            "why_bad": "Overreaching causes arm fatigue, loss of power, and reduces control",
            "key_points_to_avoid": [
                "❌ DON'T reach for holds with straight arms",
                "❌ DON'T skip foot movements to reach far",
                "❌ DON'T use purely arm strength - use legs!",
                "❌ DON'T climb with elbows locked",
                "❌ DON'T rush moves - plan each movement",
            ],
            "improvement_steps": [
                "1️⃣ Move your feet UP before reaching with your arms",
                "2️⃣ Keep elbows slightly bent during movements",
                "3️⃣ Use leg power, not just arm strength",
                "4️⃣ Practice footwork to get closer to holds",
                "5️⃣ Climb steeper angles to force better positioning",
            ],
            "drill": "Feet-First Drill: For every hand move, move feet first (opposite of usual)"
        }
    }
    
    @staticmethod
    def get_advice(issues: List[str]) -> Dict:
        """Get improvement advice for detected issues."""
        advice = {
            "total_issues": len(issues) if issues else 0,
            "issues": {}
        }
        
        if not issues or "person_not_detected" in issues:
            return advice
        
        for issue in issues:
            if issue in ImprovementAdvisor.IMPROVEMENT_TIPS:
                advice["issues"][issue] = ImprovementAdvisor.IMPROVEMENT_TIPS[issue]
        
        return advice
