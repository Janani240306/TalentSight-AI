"""
Quick run script for TalentSight AI
"""
import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("🔍 TalentSight AI - Launching")
    print("=" * 60)
    print("\n📦 Checking and installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
    print("✅ Dependencies ready!")
    print("\n🚀 Starting TalentSight AI...")
    print("📱 The app will open at: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the server\n")
    os.system("streamlit run index.py")

if __name__ == "__main__":
    main()
