<img width="2374" height="1202" alt="image" src="https://github.com/user-attachments/assets/5da6dc6f-d0ec-4daa-b8ed-83cc8f019f08" /># 🩺 EarlyNT — AI-Powered Nuchal Translucency Measurement for Early Detection  

**EarlyNT** is an AI-powered tool that assists in the **early detection of chromosomal abnormalities** by automatically **segmenting** and **measuring the Nuchal Translucency (NT)** thickness from first-trimester ultrasound images.

---

## 📖 Background  

Nuchal Translucency (NT) measurement during the first trimester is a key screening test for early detection of chromosomal abnormalities such as Down syndrome. Manual measurement may vary between operators. **EarlyNT leverages deep learning to ensure higher accuracy and consistency.**

---

## ✨ Features  

- 🧠 **Deep Learning Segmentation** — Automatically detects and segments the NT region  
- 📏 **Accurate Measurement** — Calculates NT thickness in millimeters  
- 📈 **Early Detection Aid** — Supports early screening for chromosomal anomalies such as Down syndrome  
- 💻 **Easy Integration** — Can be used as a script, API, or integrated into clinical research tools  
- 📊 **Exportable Results** — Results can be exported for clinical review or archiving  

---

## 🖼️ Example Workflow  

1. **Upload** a first-trimester ultrasound image  
2. **Model automatically segments** the NT region  
3. **NT thickness is measured** in millimeters  
4. **Results are displayed or exported** for clinical review  




## ⚙️ Tech Stack  

- **Python**  
- **TensorFlow**  
- **OpenCV**  
- **Ultralytics YOLO + SAM**  
- **Streamlit** (for the user interface)  

---

## 📋 Requirements  

- Python ≥ 3.9  
- CUDA/GPU (optional for acceleration)  
- All libraries listed in `requirements.txt`  

---

## 🚀 Installation & Usage  

```bash
# 1️⃣ Clone the repository
git clone https://github.com/salmakadeeb/Automated-Nuchal-Translucency-Measurement.git
cd Automated-Nuchal-Translucency-Measurement

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit app
streamlit run app.py
