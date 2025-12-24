#!/usr/bin/env python3
"""
Playwright Integration Test Runner for Streamlit Backtesting App

This script sets up and runs comprehensive integration tests using Playwright.
It starts the Streamlit app, runs the tests, and provides detailed reporting.

Usage:
    python run_playwright_tests.py
    python run_playwright_tests.py --headed
    python run_playwright_tests.py --debug
"""

import argparse
import asyncio
import subprocess
import sys
import time
import signal
import os
from pathlib import Path
from typing import Optional

import playwright
from playwright.async_api import async_playwright


class StreamlitTestRunner:
    """Manages Streamlit app and Playwright test execution."""
    
    def __init__(self, port: int = 8501, headless: bool = True, debug: bool = False):
        self.port = port
        self.headless = headless
        self.debug = debug
        self.streamlit_process: Optional[subprocess.Popen] = None
        self.app_url = f"http://localhost:{port}"
        
    def start_streamlit_app(self) -> bool:
        """Start the Streamlit app for testing."""
        print(f"🚀 Starting Streamlit app on port {self.port}...")
        
        try:
            # Start Streamlit app
            cmd = [
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", str(self.port),
                "--server.headless", "true",
                "--server.address", "localhost",
                "--browser.gatherUsageStats", "false"
            ]
            
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            # Wait for app to start
            max_wait_time = 60  # seconds
            wait_interval = 2  # seconds
            waited_time = 0
            
            while waited_time < max_wait_time:
                if self.streamlit_process.poll() is not None:
                    # Process died
                    stdout, stderr = self.streamlit_process.communicate()
                    print(f"❌ Streamlit app failed to start!")
                    print(f"STDOUT: {stdout.decode()}")
                    print(f"STDERR: {stderr.decode()}")
                    return False
                
                # Try to connect to the app
                try:
                    import requests
                    response = requests.get(self.app_url, timeout=5)
                    if response.status_code == 200:
                        print(f"✅ Streamlit app started successfully at {self.app_url}")
                        return True
                except:
                    pass
                
                time.sleep(wait_interval)
                waited_time += wait_interval
                print(f"⏳ Waiting for app to start... ({waited_time}s)")
            
            print(f"❌ Timed out waiting for Streamlit app to start")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start Streamlit app: {str(e)}")
            return False
    
    def stop_streamlit_app(self):
        """Stop the Streamlit app."""
        if self.streamlit_process:
            print("🛑 Stopping Streamlit app...")
            try:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
                self.streamlit_process.wait()
            except:
                pass
            self.streamlit_process = None
    
    async def verify_app_accessibility(self) -> bool:
        """Verify that the app is accessible and responsive."""
        print(f"🔍 Verifying app accessibility at {self.app_url}...")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                context = await browser.new_context()
                page = await context.new_page()
                
                # Try to access the app
                try:
                    await page.goto(self.app_url, timeout=30000)
                    await page.wait_for_selector("[data-testid='stApp']", timeout=20000)
                    
                    # Check for main title
                    title = page.locator("text=Stock Backtesting System")
                    await title.wait_for(state="visible", timeout=10000)
                    
                    print("✅ App is accessible and responsive")
                    return True
                    
                except Exception as e:
                    print(f"❌ App accessibility check failed: {str(e)}")
                    return False
                    
            finally:
                await browser.close()
    
    def run_tests(self) -> bool:
        """Run the Playwright integration tests."""
        print("🧪 Running Playwright integration tests...")
        
        # Prepare pytest command
        pytest_args = [
            sys.executable, "-m", "pytest", 
            "tests/test_playwright_integration.py",
            "-v",
            "--asyncio-mode=auto",
            f"--base-url={self.app_url}",
            "--tb=short",
            "--config-file=tests/pytest.ini"
        ]
        
        if self.debug:
            pytest_args.extend(["-s", "--log-cli-level=DEBUG"])
        else:
            pytest_args.extend(["--tb=no"])
        
        # Add HTML reporting
        pytest_args.extend([
            "--html=reports/playwright-test-report.html",
            "--self-contained-html"
        ])
        
        # Set environment variables
        env = os.environ.copy()
        env["PLAYWRIGHT_HEADLESS"] = str(self.headless).lower()
        env["PLAYWRIGHT_TIMEOUT"] = "60000"
        
        # Run tests
        try:
            result = subprocess.run(
                pytest_args,
                cwd=Path(__file__).parent.parent,
                env=env,
                capture_output=False
            )
            
            success = result.returncode == 0
            if success:
                print("✅ All tests passed!")
            else:
                print("❌ Some tests failed!")
                
            return success
            
        except Exception as e:
            print(f"❌ Failed to run tests: {str(e)}")
            return False
    
    async def run_comprehensive_test(self):
        """Run the complete test suite with setup and teardown."""
        print("🎯 Starting comprehensive integration test suite...")
        
        try:
            # Step 1: Start Streamlit app
            if not self.start_streamlit_app():
                return False
            
            # Step 2: Verify app accessibility
            if not await self.verify_app_accessibility():
                return False
            
            # Step 3: Run tests
            success = self.run_tests()
            
            if success:
                print("🎉 Integration test suite completed successfully!")
                print(f"📊 Test report available at: reports/playwright-test-report.html")
            else:
                print("💥 Integration test suite failed!")
                
            return success
            
        except KeyboardInterrupt:
            print("\n⚠️  Test suite interrupted by user")
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error during test suite: {str(e)}")
            return False
            
        finally:
            # Cleanup
            self.stop_streamlit_app()
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_streamlit_app()


def setup_signal_handlers(runner: 'StreamlitTestRunner'):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\n⚠️  Received signal {signum}, shutting down gracefully...")
        runner.cleanup()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Playwright integration tests for Streamlit app")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit app")
    parser.add_argument("--headed", action="store_true", help="Run tests in headed mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-setup", action="store_true", help="Skip Playwright browser setup")
    
    args = parser.parse_args()
    
    # Create reports directory
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Install Playwright browsers if needed
    if not args.no_setup:
        print("🔧 Installing Playwright browsers...")
        try:
            subprocess.run([
                sys.executable, "-m", "playwright", "install", "chromium"
            ], check=True)
        except subprocess.CalledProcessError:
            print("⚠️  Failed to install Playwright browsers, continuing anyway...")
    
    # Create test runner
    runner = StreamlitTestRunner(
        port=args.port,
        headless=not args.headed,
        debug=args.debug
    )
    
    # Setup signal handlers
    setup_signal_handlers(runner)
    
    try:
        # Run tests
        success = asyncio.run(runner.run_comprehensive_test())
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()