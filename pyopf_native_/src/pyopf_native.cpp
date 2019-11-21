/*
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
*/
#include "pyopf_native.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <cstddef>

#include <libopfcpp/OPF.hpp>
#include <libopfcpp/util.hpp>

namespace py = pybind11;

using namespace pybind11::literals;
using uchar = unsigned char;


std::function<py::array(py::array, double)> qGenFunc(std::function<double(double, double)> Ops)
{

    // Return an anonymous function
    return [Ops](py::array_t<double> x, double q) {
        // get the numpy array input parameter
        py::buffer_info infox = x.request();
        // access array contents
        auto ptrx = static_cast<double *>(infox.ptr);
        // acquire its size
        size_t sz = infox.size;

        // make a return numpy array -- its size equals the input array's
        py::array_t<double> result = py::array_t<double>(infox);

        auto buf3r = result.request();
        // access alloc'd memory pointer
        double *ptr3 = (double *)buf3r.ptr;

        // execute the function in series and store the result in the return pointer
        for (size_t idx = 0; idx < sz; idx++)
            ptr3[idx] = Ops(ptrx[idx], q);

        // result array
        return result;
    };
}

template <class T>
opf::Mat<T> asMat(py::array_t<float> &arr, bool copy=false)
{
    const py::buffer_info info = arr.request();

    // Consistency check
    if (info.ndim != 2)
        throw std::runtime_error("Incompatible array dimensions. Should be 2-Dimensional.");
    
    // Data is not necessarily uncontiguous
    size_t stride = info.strides[0] / sizeof(T);

    opf::Mat<T> data;
    size_t rows = info.shape[0], cols = info.shape[1];

    if (copy)
    {
        // Data allocation
        data = opf::Mat<T>(rows, cols);

        // Access pointers
        T* ptr = static_cast<T *>(info.ptr);
        T* out_ptr;

        // Copy data
        for (size_t i = 0; i < rows; i++)
        {
            ptr += stride;
            out_ptr = data.row(i);

            for (size_t j = 0; j < cols; j++)
                out_ptr[j] = ptr[j];
        }
    }
    else
    {
        // Just link to the pointer
        std::shared_ptr<T> ptr(static_cast<T*>(info.ptr), [](T *p) {});
        data = opf::Mat<T>(ptr, rows, cols, stride);
    }

    return std::move(data);
}

template <class T>
opf::distance_function<T> distance_adaptor(py::function dist_func)
{
    return [dist_func](const T *x, const T *y, int sz) -> T {
        std::vector<T> *dvx = new std::vector<T>(x, x + (sz * sizeof(x[0])));
        std::vector<T> *dvy = new std::vector<T>(y, y + (sz * sizeof(y[0])));

        auto capsulex = py::capsule(dvx, [](void *v) { delete reinterpret_cast<std::vector<T> *>(v); });
        auto capsuley = py::capsule(dvy, [](void *v) { delete reinterpret_cast<std::vector<T> *>(v); });

        auto dx = py::array(sz, x, capsulex);
        auto dy = py::array(sz, y, capsuley);

        py::object result_py = dist_func(dx, dy);

        return (T)result_py.cast<T>();
    };
}

opf::distance_function<float> get_distance(std::string& distance)
{
    if (distance == "cosine")
        return opf::cosine_distance<float>;
    else if (distance == "euclidean")
        return opf::euclidean_distance<float>;
    else
        return opf::euclidean_distance<float>;
}


class SupervisedFloatOpf
{
  public:
    // Constructors
    SupervisedFloatOpf(bool copy=false);
    SupervisedFloatOpf(bool precomputed, py::object &distance, bool copy=false);
    SupervisedFloatOpf(bool precomputed, std::string &distance, bool copy=false);

    // Functionalities
    void fit(py::array_t<float> &train_data, std::vector<int> &labels);
    std::vector<int> predict(py::array_t<float> &test_data);

    // Assignment operator
    SupervisedFloatOpf operator=(const SupervisedFloatOpf& other);

    bool get_precomputed() {return this->opf.get_precomputed();}

    // Serialization
    py::bytes serialize(uchar flags=0);
    static SupervisedFloatOpf unserialize(py::bytes& contents);

  private:
    opf::Mat<float> train_data;
    opf::SupervisedOPF<float> opf;

    bool copy;
};

SupervisedFloatOpf::SupervisedFloatOpf(bool copy) : copy(copy)
{
    this->opf = opf::SupervisedOPF<float>(false, opf::euclidean_distance<float>);
}

SupervisedFloatOpf::SupervisedFloatOpf(bool precomputed, py::object &distance, bool copy) : copy(copy)
{
    this->opf = opf::SupervisedOPF<float>(precomputed, distance_adaptor<float>(distance));
}

SupervisedFloatOpf::SupervisedFloatOpf(bool precomputed, std::string &distance, bool copy) : copy(copy)
{
    this->opf = opf::SupervisedOPF<float>(precomputed, get_distance(distance));
}

void SupervisedFloatOpf::fit(py::array_t<float> &train_data, std::vector<int> &labels)
{
    this->train_data = asMat<float>(train_data, copy);
    this->opf.fit(this->train_data, labels);
}

std::vector<int> SupervisedFloatOpf::predict(py::array_t<float> &test_data)
{
    opf::Mat<float> test_mat = asMat<float>(test_data);
    return this->opf.predict(test_mat);
}

SupervisedFloatOpf SupervisedFloatOpf::operator=(const SupervisedFloatOpf& other)
{
    if (this != &other)
    {
        this->train_data = other.train_data;
        this->opf = other.opf;
    }
    return *this;
}

py::bytes SupervisedFloatOpf::serialize(uchar flags)
{
    return py::bytes(this->opf.serialize(flags));
}

SupervisedFloatOpf SupervisedFloatOpf::unserialize(py::bytes& contents)
{
    SupervisedFloatOpf out;
    out.opf = opf::SupervisedOPF<float>::unserialize(contents);
    return std::move(out);
}

SupervisedFloatOpf OpfFactory(bool precomputed, std::string &distance, bool copy=false)
{
    SupervisedFloatOpf opfp(precomputed, distance, copy);
    return std::move(opfp);
}

SupervisedFloatOpf OpfFactory(bool precomputed, py::object &distance, bool copy=false)
{
    SupervisedFloatOpf opfp(precomputed, distance, copy);
    return std::move(opfp);
}


class UnsupervisedFloatOpf
{
  public:
    // Constructors
    UnsupervisedFloatOpf(int k=5, bool anomaly=false, float thresh=.1, bool precomputed=false, bool copy=false);
    UnsupervisedFloatOpf(int k, bool anomaly, float thresh, bool precomputed, py::object &distance, bool copy=false);
    UnsupervisedFloatOpf(int k, bool anomaly, float thresh, bool precomputed, std::string &distance, bool copy=false);
    UnsupervisedFloatOpf(opf::UnsupervisedOPF<float> opf, bool copy=false) : opf(opf), copy(copy) {}

    // Functionalities
    void fit(py::array_t<float> &train_data);
    std::vector<int> fit_predict(py::array_t<float> &train_data);
    std::vector<int> predict(py::array_t<float> &test_data);
    void find_best_k(py::array_t<float> &train_data, int kmin, int kmax, int step);

    // Assignment operator
    UnsupervisedFloatOpf operator=(const UnsupervisedFloatOpf& other);
    
    // Serialization
    py::bytes serialize(uchar flags=0);
    static UnsupervisedFloatOpf unserialize(py::bytes& contents);

    // Getters and setters
    int get_k() {return this->opf.get_k();}
    int get_n_clusters() {return this->opf.get_n_clusters();}
    bool get_anomaly() {return this->opf.get_anomaly();}
    float get_thresh() {return this->opf.get_thresh();}
    void set_thresh(float thresh) {this->opf.set_thresh(thresh);}
    bool get_precomputed() {return this->opf.get_precomputed();}

  private:
    opf::Mat<float> train_data;
    opf::UnsupervisedOPF<float> opf;

    bool copy;
};

UnsupervisedFloatOpf::UnsupervisedFloatOpf(int k, bool anomaly, float thresh, bool precomputed, bool copy) : copy(copy)
{
    this->opf = opf::UnsupervisedOPF<float>(k, anomaly, thresh, precomputed, opf::euclidean_distance<float>);
}

UnsupervisedFloatOpf::UnsupervisedFloatOpf(int k, bool anomaly, float thresh, bool precomputed, py::object &distance, bool copy) : copy(copy)
{
    this->opf = opf::UnsupervisedOPF<float>(k, anomaly, thresh, precomputed, distance_adaptor<float>(distance));
}

UnsupervisedFloatOpf::UnsupervisedFloatOpf(int k, bool anomaly, float thresh, bool precomputed, std::string &distance, bool copy) : copy(copy)
{
    this->opf = opf::UnsupervisedOPF<float>(k, anomaly, thresh, precomputed, get_distance(distance));
}

void UnsupervisedFloatOpf::fit(py::array_t<float> &train_data)
{
    this->train_data = asMat<float>(train_data, this->copy);
    this->opf.fit(this->train_data);
}

std::vector<int> UnsupervisedFloatOpf::fit_predict(py::array_t<float> &train_data)
{
    this->train_data = asMat<float>(train_data, this->copy);
    return this->opf.fit_predict(this->train_data);
}

std::vector<int> UnsupervisedFloatOpf::predict(py::array_t<float> &test_data)
{
    opf::Mat<float> test_mat = asMat<float>(test_data);
    return this->opf.predict(test_mat);
}

UnsupervisedFloatOpf UnsupervisedFloatOpf::operator=(const UnsupervisedFloatOpf& other)
{
    if (this != &other)
    {
        this->train_data = other.train_data;
        this->opf = other.opf;
    }
    return *this;
}

void UnsupervisedFloatOpf::find_best_k(py::array_t<float> &train_data, int kmin, int kmax, int step)
{
    this->train_data = asMat<float>(train_data, this->copy);
    this->opf.find_best_k(this->train_data, kmin, kmax, step);
}

py::bytes UnsupervisedFloatOpf::serialize(uchar flags)
{
    return py::bytes(this->opf.serialize(flags));
}

UnsupervisedFloatOpf UnsupervisedFloatOpf::unserialize(py::bytes& contents)
{
    // std::string tmp = contents;
    UnsupervisedFloatOpf out;
    out.opf = opf::UnsupervisedOPF<float>::unserialize(contents);
    return std::move(out);
}


UnsupervisedFloatOpf UOpfFactory(int k, bool anomaly, float thresh, bool precomputed, std::string &distance, bool copy=false)
{
    UnsupervisedFloatOpf opfp(k, anomaly, thresh, precomputed, distance, copy);
    return std::move(opfp);
}

UnsupervisedFloatOpf UOpfFactory(int k, bool anomaly, float thresh, bool precomputed, py::object &distance, bool copy=false)
{
    UnsupervisedFloatOpf opfp(k, anomaly, thresh, precomputed, distance, copy);
    return std::move(opfp);
}

PYBIND11_MODULE(pyopf_native, m)
{
    py::class_<SupervisedFloatOpf>(m, "SupervisedFloatOpf")
        .def(py::init())
        .def("fit", &SupervisedFloatOpf::fit)
        .def("predict", &SupervisedFloatOpf::predict)
        .def("get_precomputed", &SupervisedFloatOpf::get_precomputed)
        .def("serialize", &SupervisedFloatOpf::serialize)
        .def_static("unserialize", static_cast<SupervisedFloatOpf (*)(py::bytes &)>(&SupervisedFloatOpf::unserialize), py::return_value_policy::reference)
        .def_static("SupervisedOpfFloatFactory", static_cast<SupervisedFloatOpf (*)(bool, std::string &, bool)>(&OpfFactory), py::return_value_policy::reference)
        .def_static("SupervisedOpfFloatFactory", static_cast<SupervisedFloatOpf (*)(bool, py::object &, bool)>(&OpfFactory), py::return_value_policy::reference);
    
    py::class_<UnsupervisedFloatOpf>(m, "UnsupervisedFloatOpf")
        .def(py::init())
        .def("fit", &UnsupervisedFloatOpf::fit)
        .def("fit_predict", &UnsupervisedFloatOpf::fit_predict)
        .def("predict", &UnsupervisedFloatOpf::predict)
        .def("find_best_k", &UnsupervisedFloatOpf::find_best_k)
        .def("get_k", &UnsupervisedFloatOpf::get_k)
        .def("get_n_clusters", &UnsupervisedFloatOpf::get_n_clusters)
        .def("get_anomaly", &UnsupervisedFloatOpf::get_anomaly)
        .def("get_thresh", &UnsupervisedFloatOpf::get_thresh)
        .def("set_thresh", &UnsupervisedFloatOpf::set_thresh)
        .def("get_precomputed", &UnsupervisedFloatOpf::get_precomputed)
        .def("serialize", &UnsupervisedFloatOpf::serialize)
        .def_static("unserialize", static_cast<UnsupervisedFloatOpf (*)(py::bytes &)>(&UnsupervisedFloatOpf::unserialize), py::return_value_policy::reference)
        .def_static("UnsupervisedOpfFloatFactory", static_cast<UnsupervisedFloatOpf (*)(int, bool, float, bool, std::string &, bool)>(&UOpfFactory), py::return_value_policy::reference)
        .def_static("UnsupervisedOpfFloatFactory", static_cast<UnsupervisedFloatOpf (*)(int, bool, float, bool, py::object &, bool)>(&UOpfFactory), py::return_value_policy::reference);
}

