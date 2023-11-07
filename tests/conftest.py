import os
import time

import pytest

pytest_plugins = "tests.fixtures"

def pytest_collection_modifyitems(config, items):
    for item in items:
        if "apps" in item.fspath.strpath:
            item.add_marker(pytest.mark.apps)
               
def pytest_addoption(parser):
    parser.addoption("--seed", action="store", default=None, help="Seed for random number generators")
    
@pytest.fixture(scope="session")
def GLOBAL_SEED(pytestconfig):
    seed_value = pytestconfig.getoption("seed")
    if seed_value is not None:
        seed_value = int(seed_value)
    else:
        seed_value = int(time.time()) ^ (os.getpid() << 16) & 2**32-1
    print(f"Using seed: {seed_value}")
    return seed_value
