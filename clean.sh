

rm -r build dist

pip uninstall -y pyopf

python setup.py build_ext -i
python setup.py install


