# Unit Tests

```bash
pip install pytest
```

Use `-s` to enable printing to screen during the tests.

```bash
# test all
pytest -s tests

# test one file
pytest -s tests/test_algorithms.py

# test one function
pytest -s tests/test_algorithms.py::test_admm
```