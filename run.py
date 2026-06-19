"""
Quick run script for AI Recruiter
"""
import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("🎯 AI-Powered Intelligent Recruiter System")
    print("=" * 60)
    print("\n📦 Checking and installing dependencies...")
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
    
    print("✅ Dependencies installed successfully!")
    print("\n🚀 Launching the application...")
    print("📱 The app will open in your default browser at: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the server when done\n")
    
    # Run Streamlit app (Updated to run index.py!)
    os.system("streamlit run index.py")

if __name__ == "__main__":
    main()