import streamlit as st
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from io import BytesIO
from fpdf import FPDF
import random
try:
    from route_planner import SafeRouteAnalyzer
except ImportError:
    SafeRouteAnalyzer = None

# Reset the default graph to avoid warnings
tf.reset_default_graph()

# -------------------------------------------------
# Page configuration & custom fonts
# -------------------------------------------------
st.set_page_config(
    page_title="ClimbAssist AI",
    page_icon="🧗",
    layout="wide",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-bottom: 1rem;
        margin-top: 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(139, 92, 246, 0.5);
        box-shadow: 0 8px 32px 0 rgba(139, 92, 246, 0.2);
    }
    
    .gear-list-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 1.5rem;
        margin-top: 1rem;
        color: #ffffff;
        font-size: 1rem;
        line-height: 1.8;
    }
    
    .value-indicator {
        display: inline-block;
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
        color: #fff;
        border-radius: 12px;
        padding: 4px 16px;
        font-size: 0.9rem;
        margin-left: 8px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
    
    .upload-zone {
        border: 2px dashed rgba(139, 92, 246, 0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: rgba(139, 92, 246, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: rgba(139, 92, 246, 0.8);
        background: rgba(139, 92, 246, 0.1);
    }
    
    .upload-icon {
        font-size: 3rem;
        display: block;
        margin-bottom: 1rem;
    }
    
    .status-bar {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .mistake-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        border-left: 4px solid #ef4444;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        color: #fecaca;
        font-size: 0.95rem;
        animation: slideIn 0.5s ease;
    }
    
    .recommend-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.15) 100%);
        border-left: 4px solid #22c55e;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        color: #bbf7d0;
        font-size: 0.95rem;
        animation: slideIn 0.5s ease;
    }
    
    .footer-status {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        text-align: center;
        color: #c4b5fd;
        font-size: 0.9rem;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Streamlit specific overrides */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
    }
    
    h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #e0e7ff !important;
        font-weight: 600 !important;
    }
    
    /* Remove default Streamlit padding/margins */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    
    /* Remove column padding at top */
    div[data-testid="column"] {
        padding-top: 0 !important;
    }
    
    /* Remove all container padding */
    .main .block-container {
        padding-top: 0rem !important;
    }
    
    /* Hide empty spaces in columns */
    div[data-testid="column"] > div:empty {
        display: none !important;
    }
    
    /* Adjust file uploader styling */
    .stFileUploader {
        margin-top: 0 !important;
    }
    
    /* Remove extra spacing from selectbox */
    .stSelectbox {
        margin-top: 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# -------------------------------------------------
# Main Header
# -------------------------------------------------
st.markdown("""
    <div style='text-align: center; margin-bottom: 0.5rem; margin-top: 0; padding-top: 1rem;'>
        <h1 style='font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.3rem;'>
            🧗 ClimbAssist AI
        </h1>
        <p style='font-size: 1.2rem; color: #a5b4fc; font-weight: 500; margin-bottom: 0;'>
            AI-Powered Climbing Analysis & Gear Optimization
        </p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Tab Navigation
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "⚙ Gear Optimizer",
    "🎥 Movement Analyzer",
    "🗺 Safe Route Finder"
])

# -------------------------------------------------
# TAB 1: GEAR LOAD OPTIMIZER
# -------------------------------------------------
with tab1:
    left, right = st.columns([1.1, 2.2], gap="large")
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## ⚙ Gear Load Optimizer")
        st.markdown("<p style='color: #a5b4fc; margin-bottom: 1.5rem;'>Input your expedition parameters to generate an optimized gear list.</p>", unsafe_allow_html=True)
        
        # Randomized tip for gear optimization
        gear_tips = [
            "💡 <i>Pro tip: Pack 20% lighter than you think you need</i>",
            "💡 <i>Consider weather changes at high altitude</i>",
            "💡 <i>Multi-purpose gear saves weight and space</i>",
            "💡 <i>Always pack backup safety equipment</i>",
            "💡 <i>Test all gear before your expedition</i>",
            "💡 <i>Layering system is key for temperature control</i>",
            "💡 <i>Don't forget emergency communication devices</i>",
            "💡 <i>High-calorie, lightweight food is essential</i>"
        ]
        random_gear_tip = random.choice(gear_tips)
        st.markdown(f"<p style='color: #fbbf24; font-size: 0.9rem; margin-bottom: 1rem; font-style: italic;'>{random_gear_tip}</p>", unsafe_allow_html=True)

        expedition_days = st.slider(
            "Expedition Duration (days)", min_value=1, max_value=20, value=3, key="exp_days"
        )
        st.markdown(f'<span class="value-indicator">{expedition_days} days</span>', unsafe_allow_html=True)

        max_altitude = st.slider(
            "Max Altitude (m)", min_value=500, max_value=8000, value=2560, step=10, key="max_alt"
        )
        st.markdown(f'<span class="value-indicator">{max_altitude} m</span>', unsafe_allow_html=True)

        weather = st.selectbox(
            "Expected Weather", ["Clear", "Windy", "Rainy", "Snow"], key="weather"
        )
        skill = st.selectbox(
            "Climber Skill Level", ["Beginner", "Intermediate", "Advanced"], key="skill"
        )
        rock_type = st.selectbox(
            "Rock Type", ["Granite", "Limestone", "Sandstone", "Basalt"], key="rock"
        )
        
        st.markdown("<div style='margin-top: 1.5rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        import time
        optimize_btn = st.button("⚙ Optimize Gear", key="opt_btn", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if optimize_btn:
            with st.spinner("Optimizing gear..."):
                time.sleep(1.5)
            
            # Gear with usage descriptions
            gear_usage = {
                "Helmet": "Protect head from falling rocks and impacts during belaying",
                "Harness": "Essential for rope connection and weight distribution",
                "60 m Rope": "Standard length for multi-pitch routes and rappelling",
                "Quickdraws (12-15)": "Clip into bolts for lead climbing protection",
                "Belay Device (ATC/GriGri)": "Control rope for safe belaying and rappelling",
                "Carabiners (Locking)": "Connect rope to harness and build anchors",
                "Climbing Shoes": "Specialized rubber for friction and precision",
                "Chalk Bag": "Keep hands dry for better grip",
                "Portaledge / Bivy Gear": "Overnight accommodation on multi-day walls",
                "Down Jacket": "Insulation for cold temperatures at altitude",
                "High-Altitude Boots": "Insulated footwear for snow/ice conditions",
                "Gloves": "Protect hands from cold and rope burns",
                "Waterproof Shell": "Protection from rain, snow, and wind",
                "Extra Anchors / Assisted Belay Device": "Additional safety for beginners",
                "Crash Pad": "Protection for bouldering falls",
                "Ice Axe": "Self-arrest and climbing on snow/ice",
                "Crampons": "Traction on ice and hard snow",
                "Slings/Runners": "Build anchors and extend protection"
            }
            
            base_gear = [
                "Helmet",
                "Harness",
                "60 m Rope",
                "Quickdraws (12-15)",
                "Belay Device (ATC/GriGri)",
                "Carabiners (Locking)",
                "Climbing Shoes",
                "Chalk Bag"
            ]
            extra_gear = []
            if expedition_days > 7:
                extra_gear.append("Portaledge / Bivy Gear")
                extra_gear.append("Slings/Runners")
            if max_altitude > 4000:
                extra_gear.extend([
                    "Down Jacket", "High-Altitude Boots", "Gloves"
                ])
            if max_altitude > 5000:
                extra_gear.extend(["Ice Axe", "Crampons"])
            if weather in ["Rainy", "Snow"]:
                extra_gear.append("Waterproof Shell")
            if skill == "Beginner":
                extra_gear.append("Extra Anchors / Assisted Belay Device")
                extra_gear.append("Crash Pad")
            
            recommended = base_gear + extra_gear

            st.markdown('<div class="gear-list-card fade-in">', unsafe_allow_html=True)
            st.markdown("### 🧗 Recommended Gear List")
            gear_list_str = ""
            for item in recommended:
                usage = gear_usage.get(item, "Essential climbing equipment")
                st.markdown(f"{item}")
                st.markdown(f"<p style='color: #94a3b8; font-size: 0.9rem; margin-left: 1.5rem; margin-top: -0.5rem; margin-bottom: 0.8rem;'>↳ <i>{usage}</i></p>", unsafe_allow_html=True)
                time.sleep(0.15)
                gear_list_str += f"- {item}: {usage}\n"
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Randomized additional recommendations
            additional_tips = [
                "🎒 Consider weight distribution in your pack",
                "🧊 Hydration system is crucial - plan for 3L per day",
                "🔦 Bring backup headlamps and extra batteries",
                "🗺 Download offline maps before departure",
                "⚡ Solar charger recommended for multi-day trips",
                "🧭 GPS device + compass as backup navigation",
                "🏕 Lightweight emergency shelter is essential",
                "🍫 Pack high-energy snacks for quick boosts"
            ]
            random_additional_tips = random.sample(additional_tips, 3)
            st.markdown('<div class="glass-card fade-in" style="margin-top: 1.5rem; background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(99, 102, 241, 0.15)); border: 2px solid rgba(139, 92, 246, 0.3);">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #c4b5fd; margin-bottom: 1rem;">📋 Additional Recommendations</h3>', unsafe_allow_html=True)
            for tip in random_additional_tips:
                st.markdown(f"<div style='margin-bottom: 0.6rem; color: #cbd5e1;'>• {tip}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # PDF download
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', size=16)
            pdf.cell(200, 10, txt="ClimbAssist AI - Recommended Gear List", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Arial", size=11)
            pdf.cell(200, 8, txt=f"Expedition: {expedition_days} days | Altitude: {max_altitude}m | Weather: {weather} | Skill: {skill}", ln=True)
            pdf.ln(5)
            
            for item in recommended:
                usage = gear_usage.get(item, "Essential climbing equipment")
                pdf.set_font("Arial", 'B', size=12)
                pdf.cell(200, 8, txt=f"* {item}", ln=True)
                pdf.set_font("Arial", 'I', size=10)
                pdf.cell(200, 6, txt=f"   {usage}", ln=True)
                pdf.ln(2)
            
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            st.download_button(
                label="📥 Download Gear List (PDF)",
                data=pdf_bytes,
                file_name="climbassist_gear_list.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    right = st.container()
    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h2 style='font-weight:700; margin-bottom: 0.5rem;'>🎥 Climb Movement Analyzer</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#a5b4fc; font-size:1.05rem; margin-bottom: 1.5rem;'>Upload a climbing video to receive AI-powered movement analysis and personalized feedback.</p>", unsafe_allow_html=True)
        
        # Randomized video analysis tip
        video_tips = [
            "🎯 <i>Best results: Side-angle view with full body visible</i>",
            "🎯 <i>Ensure good lighting for accurate pose detection</i>",
            "🎯 <i>Capture 10-30 seconds of continuous climbing</i>",
            "🎯 <i>Avoid zooming in/out during recording</i>",
            "🎯 <i>Keep camera stable for better analysis</i>",
            "🎯 <i>Record from 3-5 meters away for optimal view</i>",
            "🎯 <i>Include warm-up movements for complete analysis</i>",
            "🎯 <i>Film challenging sections for detailed feedback</i>"
        ]
        random_video_tip = random.choice(video_tips)
        st.markdown(f"<p style='color: #34d399; font-size: 0.9rem; margin-bottom: 1rem; font-style: italic;'>{random_video_tip}</p>", unsafe_allow_html=True)
        
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        st.markdown('<span class="upload-icon">📤</span>', unsafe_allow_html=True)
        st.markdown('''
            <div style="text-align: center;">
                <p style="color: #c4b5fd; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
                    Upload Your Climbing Video
                </p>
                <p style="color: #a5b4fc; font-size: 0.95rem; margin-bottom: 0.5rem;">
                    Drag and drop or click to browse
                </p>
                <p style="color: #818cf8; font-size: 0.85rem;">
                    Supported formats: MP4, MPEG4 • Max size: 200 MB
                </p>
            </div>
        ''', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Upload Climbing Video", type=["mp4", "mpeg4"], accept_multiple_files=False, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_video is None:
            st.markdown('''
                <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; margin-top: 2rem; border: 1px solid rgba(139, 92, 246, 0.3);">
                    <h3 style="color: #c4b5fd; margin-bottom: 1rem; font-size: 1.1rem;">📋 How it works:</h3>
                    <ul style="color: #a5b4fc; line-height: 1.8; font-size: 0.95rem;">
                        <li>Upload a video of your climbing session</li>
                        <li>Our AI analyzes your body movements and technique</li>
                        <li>Receive detailed feedback on form and positioning</li>
                        <li>Get personalized recommendations to improve</li>
                    </ul>
                    <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border-left: 3px solid #3b82f6;">
                        <p style="color: #93c5fd; margin: 0; font-size: 0.9rem;">
                            💡 <b>Tip:</b> For best results, use a video with clear view of your full body and good lighting.
                        </p>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        if uploaded_video is not None:
            if cv2 is None:
                st.error("OpenCV (cv2) is not installed. Please run 'pip install opencv-python' in your terminal and restart the app.")
            else:
                st.markdown(f"<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid rgba(139, 92, 246, 0.3);'><span style='color:#c4b5fd;'>📄 <b>File:</b> {uploaded_video.name} | <b>Size:</b> {round(uploaded_video.size/1024/1024,2)} MB</span></div>", unsafe_allow_html=True)
                # Save uploaded video to temp file
                video_bytes = uploaded_video.read()
                temp_video_path = "temp_uploaded_video.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(video_bytes)

                # Load MoveNet model
                with st.spinner("🤖 Loading AI Model..."):
                    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
                    movenet = model.signatures['serving_default']

                # Helper to run pose estimation on a frame
                def detect_pose(frame):
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (192,192))
                    input_img = tf.convert_to_tensor(img, dtype=tf.int32)
                    input_img = tf.expand_dims(input_img, axis=0)
                    outputs = movenet(input_img)
                    keypoints = outputs['output_0'].numpy()[0,0,:,:]
                    return keypoints

                # Analyze video frames
                cap = cv2.VideoCapture(temp_video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Issue counters for aggregation
                issue_counts = {
                    'wide_stance': 0,
                    'hips_away': 0,
                    'poor_hand_use': 0,
                    'unstable_knees': 0,
                    'overreaching': 0
                }
                
                analyzed_frames = 0
                no_person_frames = 0
                valid_frames = 0
                
                # Sample every 5th frame to speed up analysis
                sample_rate = 5
                progress = st.progress(0, text="🔍 Analyzing climbing technique with AI...")
                
                for i in range(0, frame_count, sample_rate):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    keypoints = detect_pose(frame)
                    analyzed_frames += 1
                    
                    # Check if person detected (nose confidence > 0.3)
                    if keypoints[0,2] < 0.3:
                        no_person_frames += 1
                        continue
                    
                    valid_frames += 1
                    
                    # Extract keypoints
                    left_ankle = keypoints[15]
                    right_ankle = keypoints[16]
                    left_knee = keypoints[13]
                    right_knee = keypoints[14]
                    left_hip = keypoints[11]
                    right_hip = keypoints[12]
                    left_shoulder = keypoints[5]
                    right_shoulder = keypoints[6]
                    left_wrist = keypoints[9]
                    right_wrist = keypoints[10]
                    left_elbow = keypoints[7]
                    right_elbow = keypoints[8]
                    
                    # Issue 1: Wide stance (feet too far apart)
                    if abs(left_ankle[0] - right_ankle[0]) > 0.35 and left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
                        issue_counts['wide_stance'] += 1
                    
                    # Issue 2: Hips too far from wall
                    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
                    if avg_hip_y > 0.65 and left_hip[2] > 0.3 and right_hip[2] > 0.3:
                        issue_counts['hips_away'] += 1
                    
                    # Issue 3: Poor hand usage (low confidence on one hand)
                    if (left_wrist[2] < 0.25 or right_wrist[2] < 0.25) and (left_wrist[2] > 0.1 or right_wrist[2] > 0.1):
                        issue_counts['poor_hand_use'] += 1
                    
                    # Issue 4: Unstable knee alignment
                    if left_knee[2] > 0.3 and right_knee[2] > 0.3:
                        knee_distance = abs(left_knee[0] - right_knee[0])
                        hip_distance = abs(left_hip[0] - right_hip[0])
                        if knee_distance > hip_distance * 1.5:
                            issue_counts['unstable_knees'] += 1
                    
                    # Issue 5: Overreaching (arms too extended)
                    if left_elbow[2] > 0.3 and right_elbow[2] > 0.3:
                        left_arm_extended = abs(left_shoulder[1] - left_elbow[1]) > 0.2
                        right_arm_extended = abs(right_shoulder[1] - right_elbow[1]) > 0.2
                        if left_arm_extended or right_arm_extended:
                            issue_counts['overreaching'] += 1
                    
                    if i % max(sample_rate * 10, 1) == 0:
                        progress.progress(min(i/frame_count, 0.99), text="🔍 Analyzing climbing technique with AI...")
                
                cap.release()
                progress.progress(1.0, text="✅ Analysis complete!")
                progress.empty()
                
                # Generate mistakes and recommendations based on thresholds
                mistakes = []
                recommendations = []
                
                threshold = valid_frames * 0.15  # Issue appears in >15% of frames
                
                if issue_counts['wide_stance'] > threshold:
                    mistakes.append(f"Wide stance detected in {int((issue_counts['wide_stance']/valid_frames)*100)}% of movements")
                    recommendations.append("Keep feet closer together and focus on precise foot placements. Use smaller, controlled steps.")
                
                if issue_counts['hips_away'] > threshold:
                    mistakes.append(f"Hips positioned away from wall in {int((issue_counts['hips_away']/valid_frames)*100)}% of the climb")
                    recommendations.append("Keep hips close to the wall to maintain balance and conserve energy. Engage your core.")
                
                if issue_counts['poor_hand_use'] > threshold:
                    mistakes.append(f"Inconsistent hand usage detected in {int((issue_counts['poor_hand_use']/valid_frames)*100)}% of frames")
                    recommendations.append("Use all available handholds. Plan your sequence before each move and maintain three points of contact.")
                
                if issue_counts['unstable_knees'] > threshold:
                    mistakes.append(f"Knee instability observed in {int((issue_counts['unstable_knees']/valid_frames)*100)}% of movements")
                    recommendations.append("Keep knees aligned with your body. Avoid letting them collapse inward or flare out excessively.")
                
                if issue_counts['overreaching'] > threshold:
                    mistakes.append(f"Overreaching detected in {int((issue_counts['overreaching']/valid_frames)*100)}% of reaches")
                    recommendations.append("Avoid overreaching with fully extended arms. Move your feet up first to reduce strain and improve control.")
                
                # Add general feedback if no major issues
                if not mistakes:
                    mistakes.append("No significant technique issues detected!")
                    recommendations.append("Your climbing technique looks solid. Continue practicing and challenging yourself with harder routes.")
                
                # AI-Based Gear Recommendations from Video Analysis
                detected_skill_level = "Advanced"
                climbing_style = "Sport Climbing"
                rope_usage_detected = False
                dynamic_movement_score = 0
                
                # Analyze climbing difficulty based on issues
                total_issues = sum(issue_counts.values())
                issue_percentage = (total_issues / valid_frames * 100) if valid_frames > 0 else 0
                
                if issue_percentage > 40:
                    detected_skill_level = "Beginner"
                    climbing_style = "Indoor/Gym Climbing"
                elif issue_percentage > 20:
                    detected_skill_level = "Intermediate"
                    climbing_style = "Sport Climbing"
                else:
                    detected_skill_level = "Advanced"
                    climbing_style = random.choice(["Sport Climbing", "Trad Climbing", "Multi-Pitch"])
                
                # Detect dynamic movements
                if issue_counts['overreaching'] > threshold:
                    dynamic_movement_score = issue_counts['overreaching'] / valid_frames * 100
                
                # Generate AI-recommended gear based on analysis
                ai_gear_recommendations = {
                    "essential": [],
                    "safety": [],
                    "performance": [],
                    "optional": []
                }
                
                # Skill-based recommendations
                if detected_skill_level == "Beginner":
                    ai_gear_recommendations["essential"] = [
                        ("Assisted Belay Device (Grigri)", "Provides auto-locking for safer belaying - essential for beginners"),
                        ("Dynamic Rope 10-10.5mm", "Thicker rope easier to handle and more forgiving on mistakes"),
                        ("Full-Body Harness or Wide-Padded Harness", "Extra comfort and safety for learning proper technique")
                    ]
                    ai_gear_recommendations["safety"] = [
                        ("Helmet with MIPS Technology", "Superior protection during learning phase with multi-directional impact protection"),
                        ("Pre-Built Quickdraws (12cm)", "Easier to clip for beginners, reduces fumbling"),
                        ("Locking Carabiners (3-4 HMS)", "Essential for building safe anchor systems")
                    ]
                    ai_gear_recommendations["performance"] = [
                        ("Moderate Climbing Shoes", "Comfortable shoes with moderate downturn for technique building"),
                        ("Large Chalk Bag with Belt", "Easy access and stable positioning")
                    ]
                    
                elif detected_skill_level == "Intermediate":
                    ai_gear_recommendations["essential"] = [
                        ("Standard Belay Device (ATC-Guide)", "Versatile device for lead and top-rope with guide mode"),
                        ("Dynamic Rope 9.5-10mm", "Balanced rope for sport climbing with good handling"),
                        ("Adjustable Leg-Loop Harness", "Versatile harness for different seasons and clothing layers")
                    ]
                    ai_gear_recommendations["safety"] = [
                        ("Lightweight Climbing Helmet", "Balance of protection and weight for longer climbs"),
                        ("Alpine Quickdraws (12-18cm mix)", "Variety for different clipping positions"),
                        ("Locking Carabiners (2-3 D-shape)", "Efficient shapes for most applications")
                    ]
                    ai_gear_recommendations["performance"] = [
                        ("Performance Climbing Shoes", "Downturned shoes for overhangs and steeper terrain"),
                        ("Chalk Bag + Liquid Chalk Combo", "Better grip management for longer routes"),
                        ("60-70m Dynamic Rope", "Standard length for most sport climbing routes")
                    ]
                    
                else:  # Advanced
                    ai_gear_recommendations["essential"] = [
                        ("Micro Belay Device or GriGri+", "Lightweight and efficient for advanced techniques"),
                        ("Skinny Dynamic Rope 8.9-9.4mm", "Lightweight for redpoint attempts and long routes"),
                        ("Minimalist Harness", "Lightweight design for performance and multi-pitch efficiency")
                    ]
                    ai_gear_recommendations["safety"] = [
                        ("Ultra-Light Helmet (<200g)", "Maximum protection with minimum weight penalty"),
                        ("Wire-Gate Quickdraws", "Lighter, less gate flutter, better in cold conditions"),
                        ("Screwgate + Wiregate Carabiner Mix", "Optimized weight-to-strength ratio")
                    ]
                    ai_gear_recommendations["performance"] = [
                        ("Aggressive Climbing Shoes", "Highly downturned for maximum performance on difficult routes"),
                        ("Competition Chalk Bag", "Minimal weight and streamlined design"),
                        ("Twin/Half Rope System", "For complex trad routes and reducing rope drag")
                    ]
                
                # Add dynamic movement specific gear
                if dynamic_movement_score > 10:
                    ai_gear_recommendations["optional"] = [
                        ("Knee Pads", "Protection for dynamic movements and knee-bar rests"),
                        ("Finger Tape", "Support for aggressive crimping and dynamic catches"),
                        ("Training Board/Hangboard", "Improve finger strength for dynamic movements")
                    ]
                
                # Add style-specific recommendations
                if climbing_style == "Trad Climbing":
                    ai_gear_recommendations["optional"].extend([
                        ("Cam Set (0.3-4)", "Essential for crack climbing protection"),
                        ("Nut Set", "Basic passive protection for trad routes"),
                        ("Alpine Draws + Slings", "Extend protection and reduce rope drag")
                    ])
                elif climbing_style == "Multi-Pitch":
                    ai_gear_recommendations["optional"].extend([
                        ("Double Ropes 60m", "Allows longer rappels and reduces rope drag"),
                        ("Approach Shoes", "For hiking to remote climbing areas"),
                        ("Lightweight Backpack 20-30L", "Carry gear on multi-pitch routes")
                    ])
                
                # Add video statistics
                duration = frame_count / fps if fps > 0 else 0

                st.markdown('<div class="status-bar">✅ Analysis Complete</div>', unsafe_allow_html=True)
                
                # Display video stats
                st.markdown(f'<div class="footer-status">📊 <b>Analysis Stats:</b> Duration: {duration:.1f}s | Total Frames: {frame_count} | Analyzed: {analyzed_frames} | Valid Detections: {valid_frames} | No Person: {no_person_frames}</div>', unsafe_allow_html=True)
                
                # Display AI Detected Info
                st.markdown('<div class="glass-card fade-in" style="margin-top: 1.5rem; background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(59, 130, 246, 0.2)); border: 2px solid rgba(139, 92, 246, 0.4);">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #c4b5fd; margin-bottom: 1rem;">🤖 AI Detection Summary</h3>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"*Skill Level:* {detected_skill_level}")
                with col2:
                    st.markdown(f"*Style:* {climbing_style}")
                with col3:
                    st.markdown(f"*Issue Rate:* {issue_percentage:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if mistakes:
                    st.markdown('<div class="glass-card fade-in" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
                    st.markdown('<h3 style="color: #fca5a5; margin-bottom: 1rem;">❌ Detected Issues</h3>', unsafe_allow_html=True)
                    for m in mistakes:
                        st.markdown(f'<div class="mistake-card">• {m}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if recommendations:
                    st.markdown('<div class="glass-card fade-in" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
                    st.markdown('<h3 style="color: #86efac; margin-bottom: 1rem;">✅ Recommendations</h3>', unsafe_allow_html=True)
                    for r in recommendations:
                        st.markdown(f'<div class="recommend-card">• {r}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # AI-Powered Gear Recommendations Section
                st.markdown('<div class="glass-card fade-in" style="margin-top: 1.5rem; background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1)); border: 2px solid rgba(251, 191, 36, 0.3);">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #fcd34d; margin-bottom: 1rem;">🎒 AI-Recommended Gear for Your Skill Level</h3>', unsafe_allow_html=True)
                st.markdown(f"<p style='color: #fde68a; font-size: 0.95rem; margin-bottom: 1.5rem;'>Based on your climbing analysis, here's personalized gear for <b>{detected_skill_level}</b> level <b>{climbing_style}</b></p>", unsafe_allow_html=True)
                
                # Essential Gear
                if ai_gear_recommendations["essential"]:
                    st.markdown("🔴 Essential Gear:")
                    for gear_name, gear_desc in ai_gear_recommendations["essential"]:
                        st.markdown(f"<div style='margin-left: 1rem; margin-bottom: 0.8rem;'><span style='color: #fbbf24; font-weight: 600;'>• {gear_name}</span><br><span style='color: #cbd5e1; font-size: 0.9rem; margin-left: 1.5rem;'>↳ {gear_desc}</span></div>", unsafe_allow_html=True)
                
                # Safety Gear
                if ai_gear_recommendations["safety"]:
                    st.markdown("🟡 Safety Gear:")
                    for gear_name, gear_desc in ai_gear_recommendations["safety"]:
                        st.markdown(f"<div style='margin-left: 1rem; margin-bottom: 0.8rem;'><span style='color: #fbbf24; font-weight: 600;'>• {gear_name}</span><br><span style='color: #cbd5e1; font-size: 0.9rem; margin-left: 1.5rem;'>↳ {gear_desc}</span></div>", unsafe_allow_html=True)
                
                # Performance Gear
                if ai_gear_recommendations["performance"]:
                    st.markdown("🟢 Performance Gear:")
                    for gear_name, gear_desc in ai_gear_recommendations["performance"]:
                        st.markdown(f"<div style='margin-left: 1rem; margin-bottom: 0.8rem;'><span style='color: #fbbf24; font-weight: 600;'>• {gear_name}</span><br><span style='color: #cbd5e1; font-size: 0.9rem; margin-left: 1.5rem;'>↳ {gear_desc}</span></div>", unsafe_allow_html=True)
                
                # Optional Gear
                if ai_gear_recommendations["optional"]:
                    st.markdown("⚪ Optional Gear (Recommended):")
                    for gear_name, gear_desc in ai_gear_recommendations["optional"]:
                        st.markdown(f"<div style='margin-left: 1rem; margin-bottom: 0.8rem;'><span style='color: #fbbf24; font-weight: 600;'>• {gear_name}</span><br><span style='color: #cbd5e1; font-size: 0.9rem; margin-left: 1.5rem;'>↳ {gear_desc}</span></div>", unsafe_allow_html=True)
                
                # Add gear budget estimate
                budget_estimates = {
                    "Beginner": "$800-$1,200",
                    "Intermediate": "$1,200-$2,000",
                    "Advanced": "$2,000-$3,500"
                }
                estimated_budget = budget_estimates.get(detected_skill_level, "$1,000-$2,000")
                
                st.markdown(f"<div style='margin-top: 1.5rem; padding: 1rem; background: rgba(59, 130, 246, 0.15); border-radius: 8px;'><span style='color: #93c5fd;'>💰 <b>Estimated Budget:</b> {estimated_budget}</span><br><span style='color: #94a3b8; font-size: 0.85rem;'>Prices vary by brand and quality. Prioritize essential safety gear first.</span></div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Randomized training exercises
                training_exercises = [
                    "🏋 *Core Strength:* Practice planks and hanging leg raises to improve stability",
                    "🧘 *Flexibility:* Daily stretching improves reach and prevents injury",
                    "💪 *Finger Strength:* Use hangboard 3x per week for grip endurance",
                    "🎯 *Footwork Drills:* Practice silent climbing to improve precision",
                    "🔄 *Route Memory:* Rehearse sequences mentally before attempting",
                    "⚖ *Balance Training:* One-leg stands and slackline improve stability",
                    "🏃 *Cardio Endurance:* Running/cycling 2-3x per week boosts stamina",
                    "🧗 *Technique:* Practice different grip types and body positions",
                    "📹 *Video Analysis:* Record yourself regularly to track progress",
                    "🎪 *Dynamic Moves:* Campus board training for explosive power"
                ]
                
                training_resources = [
                    "📚 *Resource:* Watch professional climbers on YouTube for technique tips",
                    "📚 *Resource:* Join a climbing community for motivation and tips",
                    "📚 *Resource:* Consider hiring a coach for personalized training",
                    "📚 *Resource:* Read 'The Rock Climber's Training Manual' by Anderson/Anderson",
                    "📚 *Resource:* Use climbing apps like 27 Crags or Mountain Project",
                    "📚 *Resource:* Track your sessions in a climbing journal"
                ]
                
                random_exercises = random.sample(training_exercises, 3)
                random_resource = random.choice(training_resources)
                
                st.markdown('<div class="glass-card fade-in" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #a78bfa; margin-bottom: 1rem;">💪 Suggested Training Exercises</h3>', unsafe_allow_html=True)
                for ex in random_exercises:
                    st.markdown(f"<div style='margin-bottom: 0.8rem; color: #cbd5e1;'>{ex}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<div class="glass-card fade-in" style="margin-top: 1.5rem; background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.15)); border: 2px solid rgba(34, 197, 94, 0.3);">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #6ee7b7; margin-bottom: 1rem;">📚 Learning Resource</h3>', unsafe_allow_html=True)
                st.markdown(f"<div style='color: #cbd5e1;'>{random_resource}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)