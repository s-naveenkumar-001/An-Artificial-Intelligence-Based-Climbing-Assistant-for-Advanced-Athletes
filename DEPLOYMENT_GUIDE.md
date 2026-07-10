# ClimbAssist AI - Setup & Deployment Guide

## 🏔️ Overview
ClimbAssist AI is an intelligent platform for climbing analysis and route planning using AI. It provides:
- ⚙️ **Gear Load Optimizer** - Calculate optimal equipment distribution
- 🏃 **Climbing Movement Analyzer** - AI-powered pose detection and feedback
- 🗺️ **Safe Route Finder** - Terrain analysis and pathfinding

---

## 🚀 Quick Start

### Option 1: Cloud Deployment (Streamlit Cloud - Recommended for Demo)

1. **Visit Live App:**
   - URL: https://climbassistai.streamlit.app
   - Features available: Gear Optimizer (full)
   - Features limited: Movement Analyzer, Route Finder (info messages)

2. **Deploy Your Own:**
   - Fork repository on GitHub
   - Go to https://share.streamlit.io
   - Click "Deploy an app"
   - Select your forked repo
   - Main file: `app_v2.py`
   - Deploy!

---

### Option 2: Local Installation (Full Features)

#### Prerequisites
- Python 3.9+
- Git
- ~5GB disk space for ML models

#### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/bhuvan0003/Climb-Assist-AI.git
cd Climb-Assist-AI

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app_v2.py
```

The app will open at `http://localhost:8501`

---

## 📦 Dependency Configuration

### Local Deployment (`requirements.txt`)
✅ All features enabled  
✅ PyTorch, OpenCV, TensorFlow included  
✅ Full ML model support  

**Installation Time:** ~5-10 minutes  
**Disk Space:** ~3-5 GB

### Cloud Deployment (`requirements-cloud.txt`)
✅ Lightweight and fast  
✅ TensorFlow-CPU (not GPU)  
✅ Optimized for Streamlit Cloud free tier  

**Installation Time:** ~1-2 minutes  
**Disk Space:** ~1-2 GB

---

## 🎯 Feature Availability

| Feature | Local | Cloud | Notes |
|---------|-------|-------|-------|
| Gear Load Optimizer | ✅ Full | ✅ Full | Works everywhere |
| Movement Analyzer | ✅ Full | ⚠️ Limited | Requires OpenCV |
| Route Finder | ✅ Full | ⚠️ Limited | Requires PyTorch |

---

## 🔧 Troubleshooting

### Issue: "OpenCV (cv2) is not installed"
**Solution (Local):**
```bash
pip install opencv-python
streamlit run app_v2.py  # Restart
```

### Issue: "PyTorch not available"
**Solution (Local):**
```bash
# For Windows
pip install torch torchvision

# For macOS/Linux
pip install torch torchvision torchaudio

streamlit run app_v2.py  # Restart
```

### Issue: ImportError on Cloud
**Why:** Cloud environment has limited resources  
**Solution:** This is expected behavior. Use local installation for full features.

### Issue: Video Takes Too Long to Process
**Solution:**
- Ensure good lighting in video
- Use 10-30 second videos
- Local processing is faster than cloud

---

## 📝 Configuration

### Local Config (`~/.streamlit/config.toml`)
```toml
[theme]
primaryColor = "#8b5cf6"
backgroundColor = "#0f0f23"
secondaryBackgroundColor = "#1a1a3f"

[server]
maxUploadSize = 1000
```

---

## 🚀 Performance Tips

### Local (Recommended for Analysis)
1. Use **GPU** if available: CUDA/Metal acceleration
2. Process batch videos for research
3. Export results for presentations

### Cloud (Recommended for Demo)
1. Use Gear Optimizer for quick calculations
2. Share app link with teammates
3. Perfect for presentations

---

## 📊 Model Information

| Component | Model | Size | Speed |
|-----------|-------|------|-------|
| Pose Detection | MoveNet Lightning | ~4MB | Real-time |
| Terrain Segmentation | DeepLabV3+ | ~150MB | 100ms per frame |
| Depth Estimation | MiDaS v2.1 | ~150MB | 200ms per frame |
| Pathfinding | A* Algorithm | - | <1s |

---

## 🔐 Security & Privacy

- No data stored on servers
- Videos processed locally (cloud) or in-memory (local)
- Temporary files deleted after processing
- No telemetry or tracking

---

## 📚 API Usage (For Developers)

### Import Core Modules Locally

```python
from pose_visualizer import PoseVisualizer
from route_planner import SafeRouteAnalyzer

# Pose Analysis
visualizer = PoseVisualizer()
analyzed_keypoints = visualizer.draw_keypoints(frame, keypoints)

# Route Planning
analyzer = SafeRouteAnalyzer()
safe_path = analyzer.compute_route(video_path)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open Pull Request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 📧 Support

For issues, questions, or suggestions:
- GitHub Issues: https://github.com/bhuvan0003/Climb-Assist-AI/issues
- Email: bhuvi0612@outlook.com

---

**Last Updated:** February 13, 2026  
**Version:** 2.0  
**Status:** ✅ Fully Functional
