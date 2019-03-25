"""
Copyright 2019 PyOPF Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from setuptools import setup
import sys
import subprocess
import os
from distutils.core import setup, Extension
from setuptools import setup, find_packages
from distutils.core import setup, Extension
from distutils import sysconfig


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        try:
            import pybind11
        except ImportError:
            if subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
                raise RuntimeError('pybind11 install failed.')

        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

cpp_args = ['-std=c++1y', '-O3', '-fopenmp']
link_args= []

ext_modules = [
    Extension(
    'pyopf_native',
    ['pyopf_native_/src/pyopf_native.cpp'],
    language='c++',
    extra_compile_args = cpp_args,
    extra_link_args = link_args
    ),
]

setup(
    name='pyopf',
    version='0.0.1',
    include_dirs=['pybind11/include', 'pyopf_native_/include', 'pyopf_native_/LibOPFcpp/include'],
    author='Marcos Cleison and Contributors',
    author_email='marcoscleison.unit@gmail.com',
    description='Pyton bind for libOPFcpp',
    ext_modules=ext_modules,
    packages=['pyopf']
)


