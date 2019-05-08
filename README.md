# PyOPF
Python binding for [LibOPFcpp](https://github.com/thierrypin/LibOPFcpp).

# Build
```bash
git clone --recursive https://github.com/marcoscleison/PyOPF.git
pip install pybind11
cd PyOPF
python setup.py build_ext -i
python setup.py install
```
To compile with clang set environment variable before build:
```bash
export CC=clang++
```

# Remove

```bash
pip uninstall pyopf
```


# Usage

```python
from pyopf import OPF
opf = OPF()
opf.fit(x,y)
pred = opf.predict(x)
```
