import os
import time
import gdown
import pytest

pytest_plugins = "tests.fixtures"

def pytest_configure(config):
    filepath = os.path.join(os.path.dirname(__file__),'data/siren.ckpt')
    if not os.path.isfile(filepath):
        gdown.download(url='https://drive.google.com/uc?id=1DnEnAClmApP-egMmIJY8Nm1wa4aPCtFE',output=filepath)
    if not os.path.isfile(filepath):
        raise RuntimeError(f'Failed to download the data file for test: {filepath}')
    else:
        print('Using',filepath)

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
