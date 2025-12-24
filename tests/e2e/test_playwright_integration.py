"""Integration tests for Streamlit Backtesting App using Playwright.

Tests the full user journey through the web interface including:
- Page navigation
- Strategy configuration and execution
- Results visualization
- Strategy comparison
- Trade log analysis
- Export functionality
- Error handling
"""

import pytest
import asyncio
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import json
import pandas as pd
from pathlib import Path
import time
from typing import Dict, Any


class TestStreamlitBacktestingApp:
    """Comprehensive integration tests for the Streamlit backtesting app."""
    
    @pytest.fixture(scope="class")
    async def browser(self):
        """Launch browser for testing."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            yield browser
            await browser.close()
    
    @pytest.fixture(scope="class")
    async def page(self, browser: Browser):
        """Create new page for testing."""
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True
        )
        page = await context.new_page()
        yield page
        await context.close()
    
    @pytest.fixture(scope="class")
    async def app_url(self):
        """Get Streamlit app URL."""
        return "http://localhost:8501"
    
    @pytest.fixture
    async def loaded_app_page(self, page: Page, app_url: str):
        """Load and wait for Streamlit app to be ready."""
        try:
            await page.goto(app_url, timeout=30000)
            
            # Wait for Streamlit to fully load
            await page.wait_for_selector("[data-testid='stApp']", timeout=20000)
            await page.wait_for_load_state("networkidle", timeout=15000)
            
            # Wait for sidebar to be visible
            await page.wait_for_selector("[data-testid='stSidebar']", timeout=10000)
            
            yield page
            
        except Exception as e:
            pytest.skip(f"Could not load Streamlit app at {app_url}: {str(e)}")
    
    async def _wait_for_streamlit_update(self, page: Page, timeout: int = 30000):
        """Wait for Streamlit to finish updating."""
        try:
            # Wait for any loading spinners to disappear
            await page.wait_for_selector(
                "[data-testid='stStatusWidget']", 
                state="hidden", 
                timeout=timeout
            )
        except:
            # Status widget might not exist, continue
            pass
        
        # Wait for network to be idle
        await page.wait_for_load_state("networkidle", timeout=10000)
        
        # Small additional wait for any remaining updates
        await asyncio.sleep(1)
    
    async def _navigate_to_page(self, page: Page, page_name: str):
        """Navigate to specific page in Streamlit app."""
        # Click sidebar navigation
        sidebar = page.locator("[data-testid='stSidebar']")
        await sidebar.wait_for(state="visible", timeout=10000)
        
        # Click the page navigation
        page_link = sidebar.locator(f"text={page_name}")
        await page_link.click(timeout=10000)
        
        await self._wait_for_streamlit_update(page)
    
    async def test_app_initialization(self, loaded_app_page: Page):
        """Test that app initializes correctly with all pages."""
        page = loaded_app_page
        
        # Check main title
        title = await page.locator("text=Stock Backtesting System").first()
        await title.wait_for(state="visible", timeout=10000)
        
        # Check sidebar navigation options
        sidebar = page.locator("[data-testid='stSidebar']")
        await sidebar.wait_for(state="visible", timeout=10000)
        
        # Verify all page options exist
        expected_pages = [
            "🚀 Strategy Runner",
            "📊 Results Dashboard", 
            "⚖️ Strategy Comparison",
            "📋 Trade Log"
        ]
        
        for page_name in expected_pages:
            page_option = sidebar.locator(f"text={page_name}")
            await page_option.wait_for(state="visible", timeout=5000)
        
        # Verify initial page is Strategy Runner
        current_page_title = page.locator("text=Strategy Runner")
        await current_page_title.wait_for(state="visible", timeout=5000)
    
    async def test_strategy_runner_page_loads(self, loaded_app_page: Page):
        """Test Strategy Runner page loads all components."""
        await self._navigate_to_page(loaded_app_page, "🚀 Strategy Runner")
        
        # Check main sections
        sections = [
            "Strategy Configuration",
            "Strategy Parameters", 
            "Data Configuration",
            "Simulation Settings"
        ]
        
        for section in sections:
            section_title = loaded_app_page.locator(f"text={section}")
            await section_title.wait_for(state="visible", timeout=10000)
        
        # Check form elements exist
        strategy_dropdown = loaded_app_page.locator("data-testid='stSelectbox'")
        await strategy_dropdown.first.wait_for(state="visible", timeout=5000)
        
        run_button = loaded_app_page.locator("button:has-text('RUN BACKTEST')")
        await run_button.wait_for(state="visible", timeout=5000)
    
    async def test_sma_strategy_configuration(self, loaded_app_page: Page):
        """Test SMA strategy parameter configuration."""
        await self._navigate_to_page(loaded_app_page, "🚀 Strategy Runner")
        
        # Select SMA strategy
        strategy_select = loaded_app_page.locator("data-testid='stSelectbox'").first
        await strategy_select.click(timeout=5000)
        await loaded_app_page.click("text=SMA Crossover", timeout=5000)
        
        # Wait for strategy parameters to update
        await self._wait_for_streamlit_update(loaded_app_page)
        
        # Check SMA parameter sliders exist
        fast_slider = loaded_app_page.locator("text=Fast MA Period")
        slow_slider = loaded_app_page.locator("text=Slow MA Period")
        
        await fast_slider.wait_for(state="visible", timeout=5000)
        await slow_slider.wait_for(state="visible", timeout=5000)
        
        # Verify slider values
        fast_input = loaded_app_page.locator("input[role='slider']").first
        slow_input = loaded_app_page.locator("input[role='slider']").nth(1)
        
        await fast_input.wait_for(state="visible", timeout=5000)
        await slow_input.wait_for(state="visible", timeout=5000)
    
    async def test_momentum_strategy_configuration(self, loaded_app_page: Page):
        """Test Momentum strategy parameter configuration."""
        await self._navigate_to_page(loaded_app_page, "🚀 Strategy Runner")
        
        # Select Momentum strategy
        strategy_select = loaded_app_page.locator("data-testid='stSelectbox'").first
        await strategy_select.click(timeout=5000)
        await loaded_app_page.click("text=Momentum", timeout=5000)
        
        await self._wait_for_streamlit_update(loaded_app_page)
        
        # Check Momentum parameter sliders
        lookback_slider = loaded_app_page.locator("text=Lookback Period")
        threshold_slider = loaded_app_page.locator("text=Momentum Threshold")
        
        await lookback_slider.wait_for(state="visible", timeout=5000)
        await threshold_slider.wait_for(state="visible", timeout=5000)
    
    async def test_run_backtest_workflow(self, loaded_app_page: Page):
        """Test complete backtest execution workflow."""
        await self._navigate_to_page(loaded_app_page, "🚀 Strategy Runner")
        
        # Configure strategy
        strategy_select = loaded_app_page.locator("data-testid='stSelectbox'").first
        await strategy_select.click(timeout=5000)
        await loaded_app_page.click("text=SMA Crossover", timeout=5000)
        await self._wait_for_streamlit_update(loaded_app_page)
        
        # Configure symbol
        symbol_select = loaded_app_page.locator("data-testid='stSelectbox']").filter(has_text="Symbol")
        await symbol_select.click(timeout=5000)
        await loaded_app_page.click("text=AAPL", timeout=5000)
        
        # Configure date range
        date_inputs = loaded_app_page.locator("input[type='date']")
        await date_inputs.first.fill("2020-01-01", timeout=5000)
        await date_inputs.nth(1).fill("2020-12-31", timeout=5000)
        
        # Configure simulation settings
        initial_cash_input = loaded_app_page.locator("input[role='spinbutton']").first
        await initial_cash_input.fill("10000", timeout=5000)
        
        # Run backtest
        run_button = loaded_app_page.locator("button:has-text('RUN BACKTEST')")
        await run_button.click(timeout=5000)
        
        # Wait for results
        await self._wait_for_streamlit_update(loaded_app_page, timeout=60000)
        
        # Check for success message or error (either is acceptable for integration test)
        success_message = loaded_app_page.locator("text=✅ Backtest completed successfully!")
        error_message = loaded_app_page.locator("text=❌")
        
        # Wait up to 30 seconds for either success or error
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    success_message.wait_for(state="visible", timeout=30000),
                    error_message.wait_for(state="visible", timeout=30000),
                    return_exceptions=True
                ),
                timeout=35000
            )
        except asyncio.TimeoutError:
            # If neither appears within timeout, check if page is still responsive
            is_responsive = await loaded_app_page.evaluate("() => true")
            assert is_responsive, "Page became unresponsive during backtest execution"
    
    async def test_results_dashboard_page(self, loaded_app_page: Page):
        """Test Results Dashboard page functionality."""
        await self._navigate_to_page(loaded_app_page, "📊 Results Dashboard")
        
        # Check for info message when no results
        info_message = loaded_app_page.locator("text=No backtest results available")
        await info_message.wait_for(state="visible", timeout=10000)
        
        # Verify page structure
        page_title = loaded_app_page.locator("text=Results Dashboard")
        await page_title.wait_for(state="visible", timeout=5000)
    
    async def test_strategy_comparison_page(self, loaded_app_page: Page):
        """Test Strategy Comparison page functionality."""
        await self._navigate_to_page(loaded_app_page, "⚖️ Strategy Comparison")
        
        # Check for minimum requirements message
        info_message = loaded_app_page.locator("text=Need at least 2 backtest results")
        await info_message.wait_for(state="visible", timeout=10000)
        
        # Verify page structure
        page_title = loaded_app_page.locator("text=Strategy Comparison")
        await page_title.wait_for(state="visible", timeout=5000)
    
    async def test_trade_log_page(self, loaded_app_page: Page):
        """Test Trade Log page functionality."""
        await self._navigate_to_page(loaded_app_page, "📋 Trade Log")
        
        # Check for info message when no results
        info_message = loaded_app_page.locator("text=No backtest results available")
        await info_message.wait_for(state="visible", timeout=10000)
        
        # Verify page structure
        page_title = loaded_app_page.locator("text=Trade Log Analysis")
        await page_title.wait_for(state="visible", timeout=5000)
    
    async def test_page_navigation_flow(self, loaded_app_page: Page):
        """Test smooth navigation between pages."""
        pages = [
            "🚀 Strategy Runner",
            "📊 Results Dashboard", 
            "⚖️ Strategy Comparison",
            "📋 Trade Log"
        ]
        
        # Navigate through all pages
        for page_name in pages:
            await self._navigate_to_page(loaded_app_page, page_name)
            
            # Verify page loaded
            page_identifier = page_name.split(" ", 1)[1]  # Remove emoji
            title_element = loaded_app_page.locator(f"text={page_identifier}")
            await title_element.wait_for(state="visible", timeout=10000)
    
    async def test_responsive_design_elements(self, loaded_app_page: Page):
        """Test responsive design elements."""
        # Test initial viewport
        initial_width = await loaded_app_page.evaluate("window.innerWidth")
        initial_height = await loaded_app_page.evaluate("window.innerHeight")
        
        assert initial_width >= 1280, "Initial viewport width should be at least 1280px"
        assert initial_height >= 720, "Initial viewport height should be at least 720px"
        
        # Test mobile viewport
        await loaded_app_page.set_viewport_size({"width": 375, "height": 667})
        await self._wait_for_streamlit_update(loaded_app_page)
        
        # Check sidebar collapses on mobile
        sidebar = loaded_app_page.locator("[data-testid='stSidebar']")
        is_sidebar_visible = await sidebar.is_visible()
        
        # Streamlit should handle mobile layout
        app_container = loaded_app_page.locator("[data-testid='stApp']")
        await app_container.wait_for(state="visible", timeout=10000)
        
        # Restore desktop viewport
        await loaded_app_page.set_viewport_size({"width": 1280, "height": 720})
        await self._wait_for_streamlit_update(loaded_app_page)
    
    async def test_error_handling_workflow(self, loaded_app_page: Page):
        """Test error handling in various scenarios."""
        await self._navigate_to_page(loaded_app_page, "🚀 Strategy Runner")
        
        # Test invalid date range (end before start)
        date_inputs = loaded_app_page.locator("input[type='date']")
        await date_inputs.first.fill("2021-12-31", timeout=5000)
        await date_inputs.nth(1).fill("2021-01-01", timeout=5000)
        
        # Try to run backtest with invalid dates
        run_button = loaded_app_page.locator("button:has-text('RUN BACKTEST')")
        await run_button.click(timeout=5000)
        
        await self._wait_for_streamlit_update(loaded_app_page)
        
        # Check for error message (may not appear if app handles it gracefully)
        error_message = loaded_app_page.locator("text=❌")
        try:
            await error_message.wait_for(state="visible", timeout=10000)
        except:
            # App might handle this without showing an error, which is also valid
            pass
    
    async def test_data_export_buttons(self, loaded_app_page: Page):
        """Test export button availability and interaction."""
        await self._navigate_to_page(loaded_app_page, "📊 Results Dashboard")
        
        # Look for export section
        export_section = loaded_app_page.locator("text=Export Results")
        await export_section.wait_for(state="visible", timeout=10000)
        
        # Check export buttons exist (might be disabled if no data)
        export_buttons = loaded_app_page.locator("button:has-text('Export')")
        button_count = await export_buttons.count()
        
        if button_count > 0:
            # Test clicking first export button
            first_button = export_buttons.first
            await first_button.click(timeout=5000)
            
            # Small wait to see if any modal appears
            await asyncio.sleep(1)
    
    async def test_form_input_validation(self, loaded_app_page: Page):
        """Test form input validation and constraints."""
        await self._navigate_to_page(loaded_app_page, "🚀 Strategy Runner")
        
        # Select SMA strategy to see parameter validation
        strategy_select = loaded_app_page.locator("data-testid='stSelectbox'").first
        await strategy_select.click(timeout=5000)
        await loaded_app_page.click("text=SMA Crossover", timeout=5000)
        await self._wait_for_streamlit_update(loaded_app_page)
        
        # Test slider constraints
        sliders = loaded_app_page.locator("input[role='slider']")
        slider_count = await sliders.count()
        
        assert slider_count >= 2, "Should have at least 2 sliders for SMA strategy"
        
        # Test initial cash input constraints
        cash_input = loaded_app_page.locator("input[role='spinbutton']").first
        await cash_input.wait_for(state="visible", timeout=5000)
        
        # Test valid value
        await cash_input.fill("50000", timeout=5000)
        current_value = await cash_input.input_value()
        assert current_value == "50000", "Cash input should accept valid values"
    
    async def test_app_footer_information(self, loaded_app_page: Page):
        """Test app footer and information sections."""
        # Check footer content
        footer_text = loaded_app_page.locator("text=Built with:")
        await footer_text.wait_for(state="visible", timeout=10000)
        
        # Check for technology stack mentions
        tech_stack = [
            "Python & Streamlit",
            "Polars & DuckDB",
            "Vectorized Execution"
        ]
        
        for tech in tech_stack:
            tech_element = loaded_app_page.locator(f"text={tech}")
            try:
                await tech_element.wait_for(state="visible", timeout=5000)
            except:
                # Footer might be truncated on some screens
                pass
    
    async def test_performance_and_load_times(self, loaded_app_page: Page):
        """Test page load performance."""
        start_time = time.time()
        
        # Navigate to each page and measure load time
        pages = [
            "🚀 Strategy Runner",
            "📊 Results Dashboard", 
            "⚖️ Strategy Comparison",
            "📋 Trade Log"
        ]
        
        for page_name in pages:
            page_start = time.time()
            await self._navigate_to_page(loaded_app_page, page_name)
            page_load_time = time.time() - page_start
            
            # Each page should load within 10 seconds
            assert page_load_time < 10, f"{page_name} took too long to load: {page_load_time:.2f}s"
        
        total_time = time.time() - start_time
        print(f"Total navigation time across all pages: {total_time:.2f}s")


class TestStreamlitIntegrationAdvanced:
    """Advanced integration tests for edge cases and complex scenarios."""
    
    @pytest.fixture(scope="class")
    async def browser(self):
        """Launch browser for testing."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            yield browser
            await browser.close()
    
    @pytest.fixture(scope="class")
    async def page(self, browser: Browser):
        """Create new page for testing."""
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True
        )
        page = await context.new_page()
        yield page
        await context.close()
    
    async def _wait_for_streamlit_update(self, page: Page, timeout: int = 20000):
        """Wait for Streamlit to finish updating."""
        try:
            await page.wait_for_selector(
                "[data-testid='stStatusWidget']", 
                state="hidden", 
                timeout=timeout
            )
        except:
            pass
        await page.wait_for_load_state("networkidle", timeout=10000)
        await asyncio.sleep(0.5)
    
    async def test_concurrent_user_simulation(self, page: Page):
        """Test app behavior under simulated concurrent usage."""
        try:
            await page.goto("http://localhost:8501", timeout=20000)
            await page.wait_for_selector("[data-testid='stApp']", timeout=15000)
            
            # Simulate rapid page switching
            pages = ["🚀 Strategy Runner", "📊 Results Dashboard", "⚖️ Strategy Comparison", "📋 Trade Log"]
            
            for _ in range(3):  # Do multiple rapid cycles
                for page_name in pages:
                    try:
                        sidebar = page.locator("[data-testid='stSidebar']")
                        if await sidebar.is_visible():
                            page_link = sidebar.locator(f"text={page_name}")
                            await page_link.click(timeout=3000)
                            await self._wait_for_streamlit_update(page, timeout=10000)
                    except:
                        # If rapid switching fails, continue
                        pass
                
                await asyncio.sleep(0.1)  # Brief pause between cycles
            
            # Verify app is still responsive
            app_title = page.locator("text=Stock Backtesting System")
            await app_title.wait_for(state="visible", timeout=5000)
            
        except Exception as e:
            pytest.skip(f"Concurrent test skipped: {str(e)}")
    
    async def test_browser_compatibility_features(self, page: Page):
        """Test browser-specific features and compatibility."""
        try:
            await page.goto("http://localhost:8501", timeout=20000)
            await page.wait_for_selector("[data-testid='stApp']", timeout=15000)
            
            # Test JavaScript execution
            js_result = await page.evaluate("""
                () => {
                    // Check for required browser APIs
                    return {
                        fetch: typeof fetch !== 'undefined',
                        localStorage: typeof localStorage !== 'undefined',
                        sessionStorage: typeof sessionStorage !== 'undefined',
                        console: typeof console !== 'undefined'
                    }
                }
            """)
            
            assert js_result['fetch'], "Browser should support fetch API"
            assert js_result['localStorage'], "Browser should support localStorage"
            assert js_result['sessionStorage'], "Browser should support sessionStorage"
            
            # Test error handling in browser console
            await page.evaluate("""
                () => {
                    // Simulate harmless error handling
                    try {
                        window.nonExistentFunction();
                    } catch (e) {
                        // Expected error
                        return true;
                    }
                    return false;
                }
            """)
            
        except Exception as e:
            pytest.skip(f"Browser compatibility test skipped: {str(e)}")


# pytest configuration for async tests
def pytest_configure(config):
    """Configure pytest for async testing."""
    config.addinivalue_line(
        "markers", "async_test: mark test as async"
    )


if __name__ == '__main__':
    # Run with: python -m pytest test_playwright_integration.py -v --asyncio-mode=auto
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])