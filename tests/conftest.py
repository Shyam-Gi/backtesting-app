"""Global pytest configuration and fixtures for backtesting system."""

import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Cleanup any test data after each test."""
    yield
    # Cleanup code here if needed
    pass


# Performance testing utilities
class PerformanceTracker:
    """Track performance during test execution."""
    
    def __init__(self):
        self.timings = {}
    
    def start_timer(self, name: str):
        """Start timing a named operation."""
        import time
        self.timings[name] = {"start": time.time()}
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        import time
        if name not in self.timings:
            raise ValueError(f"Timer '{name}' not started")
        
        self.timings[name]["end"] = time.time()
        duration = self.timings[name]["end"] - self.timings[name]["start"]
        self.timings[name]["duration"] = duration
        return duration
    
    def get_timing(self, name: str) -> float:
        """Get timing for a named operation."""
        return self.timings.get(name, {}).get("duration", 0.0)


@pytest.fixture
def perf_tracker():
    """Performance tracking fixture."""
    return PerformanceTracker()