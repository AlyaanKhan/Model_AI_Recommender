# AI Model Recommender

A beautiful, interactive Streamlit web app that recommends the best machine learning models for your dataset. Upload your CSV file, and let the app analyze your data, detect the problem type (classification or regression), and suggest top-performing models with ready-to-use code and visualizations.

## Features
- 📁 Upload your own CSV dataset
- 🎯 Automatic target column suggestion
- 🧠 Problem type detection (classification or regression)
- 🏆 Model ranking and recommendations
- 💻 Ready-to-use Python code for the best model
- 📊 Data visualizations and feature importance
- 📝 Explanations for model choices

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AlyaanKhan/Model_AI_Recommender.git
   cd Model_AI_Recommender
   ```
2. **Create and activate a virtual environment (recommended):**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install requirements:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```sh
   streamlit run model_recommender_app.py
   ```

## Usage
- Open the app in your browser (Streamlit will provide a local URL).
- Upload your CSV file using the sidebar.
- Select or create a target column.
- View model recommendations, performance metrics, and download ready-to-use code.

## Requirements
- Python 3.10 or 3.11
- See `requirements.txt` for Python package dependencies.

## License
This project is licensed under the MIT License.