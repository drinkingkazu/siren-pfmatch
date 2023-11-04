## Test running and writing

Tests are written using the [pytest](https://docs.pytest.org/en/latest/) framework. To run a barebones version of tests excluding for fundamental functions only (i.e., not testing any of the `pfmatch.apps` submodules, which takes a while), run:

```bash
cd siren-pfmatch
pytest tests/ -m 'not apps'
```

Otherwise, to run the full suite of tests, run:

```bash
cd siren-pfmatch
pytest tests/
```

To write tests, create a new file in the `tests/` directory. The directory follows the same structure as the package directory, so make sure to create the file in the correct subdirectory. For example, if you're testing a function in `siren-pfmatch/algorithms`, create your file in `tests/algorithms`. Name the file something descriptive with a `test_` prefix. Then, write your tests using pytest. A sample test file testing a function that computes simple vector norms is shown below:

```python
from (...) import l2_norm

import pytest
from tests.fixtures import rng

# pytest function
def test_l2_norm(rng):
    # test that weighted unit vectors should have norms
    # equal to the weights
    sqrt3 = np.sqrt(3)
    norm_1 = [1/sqrt3, 1/sqrt3, 1/sqrt3]

    pos_length = int(rng.random()*98) + 2
    pos = np.tile(norm_1, (pos_length, 1)) # Nx3 array

    weights = rng.random(size=(pos_length, 1))
    pos *= weights
    assert np.allclose(l2_norm(pos), weights.ravel()), \
            "expected sum of weights to equal sum of weighted unit vector norms"

    # test exception raises
    with pytest.raises(ValueError):
        l2_norm(1) # not a vector

    ...
```

Note that all test functions need to start with `test_`. See the [pytest documentation](https://docs.pytest.org/en/latest/) for more information.

### Note:
* if you are using random numbers like in the above example, import the `rng` generator from `tests.fixtures` to ensure that the tests are reproducible. Add `rng` as an argument to your test function and use it as you would a normal `np.random` generator. Pytest will automatically give it to your function. We use this to ensure that all code that uses a pseudo-random number generator has the same seed, so as to make all tests reproducable. (A [fixture](https://docs.pytest.org/en/latest/fixture.html) is a fancy name for a function that runs before all the tests and returns a value that can be used as an argument in your tests.)
* For random numbers in `pytorch`, import `torch_rng` from `tests.fixtures`, and use it as the `generator` argument to whatever random function you're using, i.e., `torch.rand(..., generator=torch_rng)`.
* Avoid running direct comparisons between float numbers. Instead, use `np.allclose` to check if two arrays are close to each other, or `torch.allclose` if you're dealing with tensors. This is because of the way that floating point numbers are stored in memory. See [this](https://docs.python.org/3/tutorial/floatingpoint.html) for more information.