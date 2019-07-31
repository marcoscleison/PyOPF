# PyOPF
Python 3 binding for [LibOPFcpp](https://github.com/thierrypin/LibOPFcpp).

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
from pyopf import OPFClassifier, OPFClustering

# Supervised classification
opf = OPFClassifier()
opf.fit(x,y)
pred = opf.predict(x)

# Unsupervised classification
opfc = OPFClustering(k = 20)

# Either just run it with a given value for k:
clusters = opfc.fit_transform(x)
# or find the best one (at a higher computational cost):
opfc.find_best_k(x)
clusters = opfc.predict(x)



```
