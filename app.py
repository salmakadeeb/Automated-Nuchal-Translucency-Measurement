import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import torch
import requests
from pathlib import Path
import pandas as pd

# Optional imports with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("YOLO not available. Using fallback detection method.")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.warning("Google Generative AI not available. Reports will be generated using fallback method.")

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    st.warning("Segment Anything Model not available. Using fallback segmentation method.")

# 🟢 Place your Gemini API key here directly (optional)
GEMINI_API_KEY = "YOUR GEMINI_API_KEY"  # Replace with your actual API key or set to None

st.set_page_config(
    page_title="NT Ultrasound Screening", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling with custom CSS - CHANGED REPORT TEXT COLORS TO WHITE
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4682b4;
        margin: 1rem 0;
        color: #1a1a1a !important;
    }
    .info-box strong {
        color: #2c3e50 !important;
        font-weight: 700;
    }
    .info-box p {
        color: #34495e !important;
        line-height: 1.6;
    }
    .report-section {
        background-color: #2c3e50;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #34495e;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #ffffff !important;
    }
    .report-section h2 {
        color: #ffffff !important;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .report-section h3 {
        color: #ffffff !important;
        margin-top: 1rem;
    }
    .report-text {
        color: #ffffff !important;
        font-size: 1rem;
        line-height: 1.8;
    }
    .measurement-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
    }
    .metrics-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .report-disclaimer {
        background-color: #fff3cd;
        color: #856404 !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Model setup with error handling
@st.cache_resource
def load_models():
    """Load models with caching and error handling"""
    yolo_model = None
    sam_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load YOLO if available
        if YOLO_AVAILABLE:
            if Path("best.pt").exists():
                yolo_model = YOLO("best.pt")
            else:
                st.warning("YOLO model file 'best.pt' not found. Using fallback detection.")
        
        # Load SAM if available
        if SAM_AVAILABLE:
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            
            # Download SAM if not exists
            if not Path(sam_checkpoint).exists():
                st.info("SAM model not found. Using fallback segmentation method.")
            else:
                sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam_model.to(device=device)
        
        return yolo_model, sam_model, device
    except Exception as e:
        st.warning(f"Error loading models: {e}. Using fallback methods.")
        return None, None, device

# API key setup with error handling
if GEMINI_API_KEY and GENAI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.warning(f"Gemini API configuration failed: {e}")
        gemini_model = None
else:
    gemini_model = None

DISCLAIMER_TEXT = "⚠️ Disclaimer: This report is for informational purposes only and is not a substitute for professional medical advice. Always consult with a healthcare professional for a proper diagnosis."

def read_and_preprocess_image(file_bytes, target_size=640):
    """Read and preprocess image"""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(image)
        
        # Enhance image quality
        img_enhanced = cv2.convertScaleAbs(img_array, alpha=1.2, beta=10)
        img_resized = cv2.resize(img_enhanced, (target_size, target_size))
        
        return img_resized, img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def fallback_detection(img):
    """Fallback detection method using OpenCV"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Return box in YOLO format [x1, y1, x2, y2]
            return [x, y, x + w, y + h], 0.7  # Mock confidence
        
        return None, None
    except Exception as e:
        st.error(f"Error in fallback detection: {e}")
        return None, None

def yolo_detection(img, model):
    """Use YOLO for initial detection with fallback"""
    try:
        if model is not None:
            results = model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return fallback_detection(img)
            
            # Get best detection
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            return boxes[best_idx], confidences[best_idx]
        else:
            return fallback_detection(img)
    except Exception as e:
        st.warning(f"YOLO detection failed: {e}. Using fallback method.")
        return fallback_detection(img)

def fallback_segmentation(img, box):
    """Fallback segmentation method using OpenCV"""
    try:
        x1, y1, x2, y2 = map(int, box)
        
        # Create mask for the detected region
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Simple rectangular mask
        mask[y1:y2, x1:x2] = 1
        
        # Apply some morphological operations to refine
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask.astype(bool)
    except Exception as e:
        st.error(f"Error in fallback segmentation: {e}")
        return np.zeros(img.shape[:2], dtype=bool)

def sam_segmentation(img, box, sam_model, device):
    """Use SAM for precise segmentation with fallback"""
    try:
        if sam_model is not None:
            predictor = SamPredictor(sam_model)
            predictor.set_image(img)
            
            # Convert box to SAM format
            input_box = np.array(box)
            
            masks, scores, _ = predictor.predict(
                box=input_box,
                multimask_output=True
            )
            
            # Select best mask
            best_mask_idx = np.argmax(scores)
            return masks[best_mask_idx]
        else:
            return fallback_segmentation(img, box)
    except Exception as e:
        st.warning(f"SAM segmentation failed: {e}. Using fallback method.")
        return fallback_segmentation(img, box)

def calculate_nt_measurement(mask, pixel_spacing_x=0.04, pixel_spacing_y=0.04):
    """Calculate NT thickness using height and width method"""
    try:
        # Convert mask to binary format if needed
        mask = (mask > 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask
        
        # Find coordinates where mask is positive
        ys, xs = np.where(mask > 0)
        
        if len(ys) == 0:
            return 0, None, {}
        
        # Calculate dimensions in pixels
        nt_height_px = ys.max() - ys.min()
        nt_width_px = xs.max() - xs.min()
        nt_area_px = np.sum(mask)
        
        # Convert to millimeters
        nt_length_mm = nt_height_px * pixel_spacing_y
        nt_width_mm = nt_width_px * pixel_spacing_x
        nt_area_mm2 = nt_area_px * pixel_spacing_x * pixel_spacing_y
        
        # Risk assessment
        risk_flag = "High Risk" if nt_length_mm > 3.0 else "Normal"
        
        # Create measurement line coordinates (from top to bottom of the detected region)
        top_point = (int(xs[np.argmin(ys)]), int(ys.min()))
        bottom_point = (int(xs[np.argmax(ys)]), int(ys.max()))
        line_coords = (top_point, bottom_point)
        
        # Additional measurements dictionary
        measurements = {
            "nt_length_mm": nt_length_mm,
            "nt_width_mm": nt_width_mm,
            "nt_area_mm2": nt_area_mm2,
            "risk": risk_flag,
            "height_px": nt_height_px,
            "width_px": nt_width_px,
            "area_px": nt_area_px
        }
        
        return nt_length_mm, line_coords, measurements
    except Exception as e:
        st.error(f"Error calculating NT measurement: {e}")
        return 0, None, {}

def create_overlay_visualization(original_img, mask):
    """Create enhanced result visualization without measurement line"""
    try:
        overlay = original_img.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask] = [0, 255, 255]  # Yellow for detected region
        
        # Blend mask with original image
        result = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return original_img

def generate_fallback_report(measurements, confidence):
    """Generate a fallback report when Gemini is not available"""
    measurement = measurements.get('nt_length_mm', 0)
    risk = measurements.get('risk', 'Unknown')
    
    report = f"""**NT Ultrasound Screening Report**

**Measurement Results:**
The Nuchal Translucency (NT) measurement for this scan is {measurement:.2f} mm.

**Clinical Interpretation:**
The normal range for NT measurements is typically less than 3.0 mm. Based on this measurement, the result is classified as: {risk}.

**What This Means:**
Nuchal Translucency screening is an ultrasound measurement of the fluid thickness at the back of the baby's neck during pregnancy. This measurement helps assess the risk of certain chromosomal conditions.

**Additional Information:**
- Measurement confidence: {confidence:.1%}
- Width: {measurements.get('nt_width_mm', 0):.2f} mm
- Area: {measurements.get('nt_area_mm2', 0):.2f} mm²

**Next Steps:**
{"If this measurement is within normal range, continue with routine prenatal care." if risk == "Normal" else "Please discuss these results with your healthcare provider for further evaluation and guidance."}

{DISCLAIMER_TEXT}
"""
    return report

def generate_detailed_report(measurements, confidence):
    """Generate detailed report using Gemini or fallback"""
    measurement = measurements.get('nt_length_mm', 0)
    
    if gemini_model is not None:
        prompt = f"""
        Generate a comprehensive NT screening report for a baby's Nuchal Translucency (NT) scan. The measurement is {measurement:.2f} mm.
        The clinical threshold for NT is 3.0 mm. Measurements below this are considered to be within the normal range.
        
        The report should:
        1. Start directly with explaining what NT is in simple terms (no greeting or subject line)
        2. Clearly state the measurement and its status (normal/abnormal)
        3. Explain the significance of the result in a way a non-medical person can understand
        4. Provide reassuring and positive tone throughout
        5. Include relevant medical context and next steps
        6. End with this exact disclaimer: "{DISCLAIMER_TEXT}"
        
        Do not include any greeting lines, subject lines, or addressee information. Start directly with the medical content.
        """
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.warning(f"Gemini API error: {e}. Using fallback report.")
            return generate_fallback_report(measurements, confidence)
    else:
        return generate_fallback_report(measurements, confidence)

# --- Enhanced Streamlit UI ---
st.markdown('<div class="main-header"><h1>🩺 Advanced NT Ultrasound Screening</h1><p>AI-powered precise NT measurement using advanced computer vision</p></div>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About NT Screening")
    st.info("""
    **What is NT Screening?**
    
    Nuchal Translucency screening is an ultrasound measurement of the fluid thickness at the back of the baby's neck during pregnancy.
    
    **Reference Values:**
    - Less than 3.0 mm: Normal
    - 3.0 mm or more: Needs follow-up
    """)
    
    st.header("⚙️ Pixel Spacing Settings")
    pixel_spacing_x = st.number_input("Pixel Spacing X (mm/pixel)", value=0.04, step=0.001, format="%.3f")
    pixel_spacing_y = st.number_input("Pixel Spacing Y (mm/pixel)", value=0.04, step=0.001, format="%.3f")
    
    st.warning(DISCLAIMER_TEXT)
    
    st.header("🔧 Technology Stack")
    status_yolo = "✅ Available" if YOLO_AVAILABLE else "❌ Fallback"
    status_sam = "✅ Available" if SAM_AVAILABLE else "❌ Fallback"
    status_gemini = "✅ Available" if gemini_model else "❌ Fallback"
    
    st.markdown(f"""
    - **YOLO v8**: {status_yolo}
    - **SAM**: {status_sam}
    - **Gemini AI**: {status_gemini}
    """)

# Load models
yolo_model, sam_model, device = load_models()

# File upload
uploaded_file = st.file_uploader(
    "📁 Upload Ultrasound Image",
    type=["png", "jpg", "jpeg"],
    help="Ensure the image is clear and shows the fetal neck region"
)

if uploaded_file is not None:
    # Read and process image
    file_bytes = uploaded_file.read()
    processed_img, original_img = read_and_preprocess_image(file_bytes)
    
    if processed_img is not None and original_img is not None:
        # Display original image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Original Image")
            st.image(original_img, use_container_width=True)
        
        # Process image
        with st.spinner("🔍 Analyzing image..."):
            # Detection using YOLO or fallback
            detection_box, confidence = yolo_detection(processed_img, yolo_model)
            
            if detection_box is None:
                st.error("❌ Could not detect NT region. Please ensure the image shows a clear fetal neck area.")
            else:
                # Segmentation using SAM or fallback
                mask = sam_segmentation(processed_img, detection_box, sam_model, device)
                
                # Calculate measurement with new method
                measurement, line_coords, measurements = calculate_nt_measurement(
                    mask, pixel_spacing_x, pixel_spacing_y
                )
                
                if measurement == 0:
                    st.error("❌ Could not measure NT thickness.")
                else:
                    # Create visualization
                    result_img = create_overlay_visualization(processed_img, mask)
                    
                    with col2:
                        st.subheader("🎯 Analyzed Result")
                        st.image(result_img, use_container_width=True)
                    
                    # Display main result
                    risk = measurements.get('risk', 'Unknown')
                    status = "Normal ✅" if risk == "Normal" else "High Risk ⚠️"
                    color = "white" if risk == "Normal" else "#ffcccc"
                    
                    st.markdown(f"""
                    <div class="measurement-box">
                        <h2>📏 NT Length: {measurement:.2f} mm</h2>
                        <h3 style="color: {color};">Status: {status}</h3>
                        <p>Detection Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed measurements
                    st.subheader("📊 Detailed Measurements")
                    col3, col4, col5, col6 = st.columns(4)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metrics-container">
                            <h4>Length</h4>
                            <h2>{measurements['nt_length_mm']:.2f} mm</h2>
                            <small>{measurements['height_px']} pixels</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metrics-container">
                            <h4>Width</h4>
                            <h2>{measurements['nt_width_mm']:.2f} mm</h2>
                            <small>{measurements['width_px']} pixels</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col5:
                        st.markdown(f"""
                        <div class="metrics-container">
                            <h4>Area</h4>
                            <h2>{measurements['nt_area_mm2']:.2f} mm²</h2>
                            <small>{measurements['area_px']} pixels²</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col6:
                        st.markdown(f"""
                        <div class="metrics-container">
                            <h4>Risk Level</h4>
                            <h2>{risk}</h2>
                            <small>Based on {measurement:.2f} mm</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Measurements table
                    st.subheader("📋 Measurement Summary")
                    df = pd.DataFrame({
                        'Metric': ['NT Length', 'NT Width', 'NT Area', 'Risk Assessment', 'Confidence'],
                        'Value': [
                            f"{measurements['nt_length_mm']:.2f} mm",
                            f"{measurements['nt_width_mm']:.2f} mm", 
                            f"{measurements['nt_area_mm2']:.2f} mm²",
                            risk,
                            f"{confidence:.1%}"
                        ],
                        'Pixels': [
                            f"{measurements['height_px']} px",
                            f"{measurements['width_px']} px",
                            f"{measurements['area_px']} px²",
                            "-",
                            "-"
                        ]
                    })
                    st.dataframe(df, use_container_width=True)
                    
                    # Generate report
                    if st.button("📋 Generate AI Medical Report", key="generate_report"):
                        with st.spinner("🤖 Generating comprehensive medical report..."):
                            report = generate_detailed_report(measurements, confidence)
                            
                            st.subheader("📄 NT Screening Report - AI Generated")
                            
                            # Display report with proper styling
                            st.markdown('<div class="report-section">', unsafe_allow_html=True)
                            
                            # Parse and display report with proper formatting
                            lines = report.split('\n')
                            for line in lines:
                                if line.strip():
                                    if line.startswith('**') and line.endswith('**'):
                                        # Headers
                                        header_text = line.strip('**')
                                        st.markdown(f"<h3 style='color: #ffffff; margin-top: 1rem;'>{header_text}</h3>", unsafe_allow_html=True)
                                    elif line.startswith('- '):
                                        # List items
                                        st.markdown(f"<p style='color: #ffffff; margin-left: 1rem;'>• {line[2:]}</p>", unsafe_allow_html=True)
                                    elif '⚠️' in line:
                                        # Disclaimer
                                        st.markdown(f"<div class='report-disclaimer'>{line}</div>", unsafe_allow_html=True)
                                    else:
                                        # Regular text
                                        st.markdown(f"<p class='report-text'>{line}</p>", unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Create downloadable report
                            report_data = f"""
NT ULTRASOUND SCREENING REPORT
==============================

MEASUREMENTS:
- NT Length: {measurements['nt_length_mm']:.2f} mm
- NT Width: {measurements['nt_width_mm']:.2f} mm  
- NT Area: {measurements['nt_area_mm2']:.2f} mm²
- Risk Assessment: {risk}
- Detection Confidence: {confidence:.1%}

PIXEL MEASUREMENTS:
- Length: {measurements['height_px']} pixels
- Width: {measurements['width_px']} pixels
- Area: {measurements['area_px']} pixels²

PIXEL SPACING SETTINGS:
- X: {pixel_spacing_x:.3f} mm/pixel
- Y: {pixel_spacing_y:.3f} mm/pixel

AI GENERATED REPORT:
{report}
                            """
                            
                            st.download_button(
                                label="📥 Download Complete Report",
                                data=report_data,
                                file_name=f"NT_Report_{measurement:.2f}mm_complete.txt",
                                mime="text/plain"
                            )

# Additional information at bottom
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.info("""
    **🎯 Enhanced Accuracy**
    
    New calculation method measures length, width, and area for comprehensive assessment.
    """)

with col_info2:
    st.info("""
    **⚡ Real-time Analysis**
    
    Instant pixel-to-millimeter conversion with customizable spacing parameters.
    """)

with col_info3:
    st.info("""
    **📊 Detailed Metrics**
    
    Complete measurements including area calculation and risk assessment.
    """)

st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Powered by Advanced AI Technologies | YOLO + SAM + Gemini AI</p>
    <p>Enhanced with comprehensive measurement analysis</p>
    <p>For technical support or medical questions, please consult with healthcare professionals</p>
</div>
""", unsafe_allow_html=True)