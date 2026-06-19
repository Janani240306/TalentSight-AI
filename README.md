# 🎯 AI-Powered Intelligent Recruiter System

An advanced candidate ranking system that goes beyond simple keyword matching. By leveraging Natural Language Processing (NLP) and Semantic Search, it parses job descriptions and candidate profiles to evaluate multi-dimensional fit.

## 🚀 Live Demo
You can host this application live for free on **Streamlit Community Cloud**:
1. Log in to [Streamlit Share](https://share.streamlit.io/) using your GitHub account.
2. Click **New app**.
3. Select this repository: `Janani240306/ai-recruiter-system`.
4. Set the branch to `main`.
5. Set the Main file path to `index.py`.
6. Click **Deploy**!

---

## 🛠️ Key Features
- **Semantic Understanding**: Uses a sentence transformer model (`all-MiniLM-L6-v2`) to capture context and meaning rather than just counting keyword matches.
- **Dynamic Weighting**: Configure ranking weights (Semantic Fit, Skill Match, Experience Match, Activity Score) dynamically in the sidebar.
- **Visual Analytics**: Interactive candidate ranking bar charts and radar plots.
- **Export Results**: Download the finalized ranking lists as CSV, detailed analysis JSON, or a fully styled HTML report.

---

## 💻 Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Janani240306/ai-recruiter-system.git
   cd ai-recruiter-system
   ```

2. **Run the startup script**:
   ```bash
   python run.py
   ```
   This script will automatically install all dependencies in `requirements.txt` and launch the Streamlit server at `http://localhost:8501`.

---

## 📦 Dependencies
Listed in `requirements.txt`:
- `streamlit`
- `pandas`
- `numpy`
- `sentence-transformers`
- `scikit-learn`
- `plotly`
- `nltk`
