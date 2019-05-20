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

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include <functional>
#include <cmath>
#include <cstdlib>
#include "libopfcpp/OPF.hpp"
#include "libopfcpp/util.hpp"
#include <iostream>
#include <memory>
#include <cstddef>
namespace py = pybind11;

using namespace pybind11::literals;


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
opf::Mat<float> AsMat(py::array_t<float> &train_data)
{
    const auto train_data_info = train_data.request();
    auto train_ptr = static_cast<float *>(train_data_info.ptr);
    std::shared_ptr<float> tptr{train_ptr};
    opf::Mat<float> train_mat(tptr, train_data.shape(0), train_data.shape(1));
    return train_mat;
}

// template <class T>
// T python_distance_adaptor(const T* a, const T* b, int size)
// {
//     T sum = 0;
//     for (size_t i = 0; i < size; i++)
//     {
//         sum += pow(a[i]-b[i], 2);
//     }
//     return (T)sqrt(sum);
// }

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
    SupervisedFloatOpf()
    {
        this->precomputed = false;
        this->distance = opf::euclidean_distance<float>;
        this->opf = opf::SupervisedOPF<float>(this->precomputed, this->distance);
    }
    SupervisedFloatOpf(bool precomputed, py::object distance)
    {
        this->precomputed = precomputed;
        this->distance = distance_adaptor<float>(distance);
        this->opf = opf::SupervisedOPF<float>(this->precomputed, this->distance);
    }

    SupervisedFloatOpf(bool precomputed, std::string &distance)
    {
        this->distance = get_distance(distance);
        this->precomputed = precomputed;
        this->opf = opf::SupervisedOPF<float>(this->precomputed, this->distance);
    }

    void fit(py::array_t<float> &train_data, std::vector<int> &labels)
    {
        const auto train_data_info = train_data.request();
        auto train_ptr = static_cast<float *>(train_data_info.ptr);

        opf::Mat<float> train_mat(train_ptr, train_data.shape(0), train_data.shape(1));
        this->opf.fit(train_mat, labels);
    }

    std::vector<int> predict(py::array_t<float> &test_data)
    {
        const auto test_data_info = test_data.request();
        auto test_ptr = static_cast<float *>(test_data_info.ptr);

        opf::Mat<float> test_mat(test_ptr, test_data.shape(0), test_data.shape(1));
        auto res = this->opf.predict(test_mat);
        return res;
    }

  private:
    opf::SupervisedOPF<float> opf;
    opf::distance_function<float> distance = nullptr;
    bool precomputed = false;
};


SupervisedFloatOpf OpfFactory(bool precomputed, std::string &distance)
{
    SupervisedFloatOpf opfp(precomputed, distance);
    return opfp;
}

SupervisedFloatOpf OpfFactory(bool precomputed, py::object &distance)
{
    SupervisedFloatOpf opfp(precomputed, distance);
    return opfp;
}

PYBIND11_MODULE(pyopf_native, m)
{
    py::class_<SupervisedFloatOpf>(m, "SupervisedFloatOpf")
        .def(py::init())
        .def("fit", &SupervisedFloatOpf::fit)
        .def("predict", &SupervisedFloatOpf::predict)
        .def_static("SupervisedOpfFloatFactory", static_cast<SupervisedFloatOpf (*)(bool , std::string &)>( &OpfFactory), py::return_value_policy::reference)
        .def_static("SupervisedOpfFloatFactory", static_cast<SupervisedFloatOpf (*)(bool , py::object &)>( &OpfFactory), py::return_value_policy::reference);
}

