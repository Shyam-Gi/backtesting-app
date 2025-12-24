#!/usr/bin/env python3
"""
Setup script for Playwright integration tests.

This script installs all necessary dependencies and sets up the testing environment.
Run this once before using the Playwright tests.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_node_version():
    """Check if Node.js is available (for Playwright browser installation)."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js {result.stdout.strip()} detected")
            return True
    except:
        pass
    
    print("⚠️  Node.js not found. Playwright browser installation may fail.")
    print("   Install Node.js from https://nodejs.org/ for best results.")
    return False


def install_python_dependencies():
    """Install Python dependencies."""
    requirements_files = [
        "requirements.txt",
        "tests/requirements-playwright.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            success = run_command(
                f"{sys.executable} -m pip install -r {req_file}",
                f"Installing from {req_file}"
            )
            if not success:
                print(f"⚠️  Failed to install {req_file}, continuing...")
        else:
            print(f"⚠️  {req_file} not found, skipping...")


def install_playwright_browsers():
    """Install Playwright browsers."""
    success = run_command(
        f"{sys.executable} -m playwright install chromium",
        "Installing Playwright Chromium browser"
    )
    
    if not success:
        print("⚠️  Browser installation failed. You may need to install manually:")
        print("   python -m playwright install chromium")
    
    return success


def create_directories():
    """Create necessary directories."""
    base_path = Path(__file__).parent.parent
    directories = ["reports", "screenshots", "videos"]
    for directory in directories:
        (base_path / directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/ directory")


def verify_installation():
    """Verify that everything is installed correctly."""
    print("\n🔍 Verifying installation...")
    
    # Check Python imports
    try:
        import playwright
        print(f"✅ Playwright {playwright.__version__} installed")
    except ImportError:
        print("❌ Playwright not installed")
        return False
    
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} installed")
    except ImportError:
        print("❌ pytest not installed")
        return False
    
    # Test Playwright browser
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("about:blank")
            browser.close()
        print("✅ Playwright browser test successful")
    except Exception as e:
        print(f"❌ Playwright browser test failed: {e}")
        return False
    
    return True


def main():
    """Main setup process."""
    print("🚀 Setting up Playwright integration test environment...")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    node_available = check_node_version()
    
    print("\n📦 Installing dependencies...")
    
    # Install Python packages
    install_python_dependencies()
    
    # Install Playwright browsers
    install_playwright_browsers()
    
    # Create directories
    create_directories()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Start your Streamlit app: streamlit run app.py")
        print("   2. Run tests: python run_playwright_tests.py")
        print("   3. Or run manually: pytest tests/test_playwright_integration.py -v --asyncio-mode=auto")
        print("\n📖 Documentation: tests/README_PLAYWRIGHT.md")
    else:
        print("\n❌ Setup completed with errors. Check the messages above.")
        if not node_available:
            print("\n💡 Try installing Node.js and running this setup again.")
        sys.exit(1)


if __name__ == "__main__":
    main()