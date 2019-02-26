# PyOPF
Python binding for LibOPFcpp.

# Build
```bash
git clone https://github.com/marcoscleison/PyOPF.git
pip install pybind11
python setup.py build_ext -i
python setup.py install
```
To compile with clang set environment variable before build:
```bash
export CC=clang++
```

# Remove

```bash
pip install pyopf
```


# Usage

```python
from pyopf import OPF
opf = OPF()
opf.fit(x,y)
pred = opf.predict(x)

```