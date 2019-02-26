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

double Xgn(double x, double q)
{

    if (q == 1.0)
    {
        return x;
    }

    return (std::pow(2.0 - q, x) - 1.0) / (1.0 - q);
}

std::function<py::array(py::array, double)> qGenFunc(std::function<double(double, double)> Ops)
{

    //Aqui eu retorno uma função anônima
    return [Ops](py::array_t<double> x, double q) {
        // pego o parâmetro com array numpy de entrada.
        py::buffer_info infox = x.request();
        //acesso o conteúdo do array do numpy.
        auto ptrx = static_cast<double *>(infox.ptr);
        // pego o tamanho
        size_t sz = infox.size;

        //crio um array do numpy de retorno. O tamanho do arrayé igual ao tamanho do array de entrada
        py::array_t<double> result = py::array_t<double>(infox);

        auto buf3r = result.request();
        // pego ponteiro na memória alocado.
        double *ptr3 = (double *)buf3r.ptr;

        // Executo a função em série e armazeno resultado no ponteiro de retorno
        for (size_t idx = 0; idx < sz; idx++)
            ptr3[idx] = Ops(ptrx[idx], q);

        //        retorno o array com resultado.
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

class SupervisedOpfFloatProxy
{
  public:
    SupervisedOpfFloatProxy()
    {
        this->precomputed = false;
        this->opf = new opf::SupervisedOPF<float>(this->precomputed, opf::cosine_distance<float>);
        //        std::cout << "Constructor" << std::endl;
        // this->distance = nullptr;
    }

    void set_distance(py::object distance)
    {

        this->distance = distance_adaptor<float>(distance);
        delete this->opf;
        this->opf = new opf::SupervisedOPF<float>(this->precomputed, this->distance);
    }

    void set_distance(opf::distance_function<float> distance)
    {
        this->distance = distance;
        delete this->opf;
        this->opf = new opf::SupervisedOPF<float>(this->precomputed, this->distance);
    }
    void set_precomputed(bool precompute)
    {
        this->precomputed = precompute;
        delete this->opf;
        this->opf = new opf::SupervisedOPF<float>(this->precomputed, this->distance);
    }

    void fit(py::array_t<float> &train_data, std::vector<int> &labels)
    {
        //        std::cout << "Fit" << std::endl;
        const auto train_data_info = train_data.request();
        auto train_ptr = static_cast<float *>(train_data_info.ptr);

        opf::Mat<float> train_mat(train_ptr, train_data.shape(0), train_data.shape(1));
        this->opf->fit(train_mat, labels);
    }

    std::vector<int> predict(py::array_t<float> &test_data)
    {
        const auto test_data_info = test_data.request();
        auto test_ptr = static_cast<float *>(test_data_info.ptr);

        opf::Mat<float> test_mat(test_ptr, test_data.shape(0), test_data.shape(1));
        auto res = this->opf->predict(test_mat);
        return res;
    }

  private:
    opf::SupervisedOPF<float> *opf;
    opf::distance_function<float> distance = nullptr;
    bool precomputed = false;
};

SupervisedOpfFloatProxy *OpfFactory(bool precomputed, std::string &distance)
{

    SupervisedOpfFloatProxy *opfp = new SupervisedOpfFloatProxy();

    if (distance == "cos")
    {

        opfp->set_distance(opf::cosine_distance<float>);
        opfp->set_precomputed(precomputed);
    }
    else if (distance == "euclidean")
    {
        opfp->set_distance(opf::euclidean_distance<float>);
        opfp->set_precomputed(precomputed);
    }
    else
    {
        opfp->set_distance(opf::euclidean_distance<float>);
        opfp->set_precomputed(precomputed);
    }
    return opfp;
}

SupervisedOpfFloatProxy *OpfFactory(bool precomputed, py::object &distance)
{
    SupervisedOpfFloatProxy *opfp = new SupervisedOpfFloatProxy();
    opfp->set_distance(distance);
    opfp->set_precomputed(precomputed);
    return opfp;
}

PYBIND11_MODULE(pyopf_native, m)
{
    // py::module m("pyopf_native", "OPFcpp natve binding.");
    m.def("Xgn", &Xgn, "Single value deformed number",
          "x"_a = 1.0, "q"_a = 0.9);

    // m.def("SupervisedOpfFloatFactory", &SupervisedOpfFloatProxy::OpfFactory, "SupervisedOpfFloatProxy Factory",
    //   "precomputed"_a = false, "distance"_a = opf::euclidean_distance);
    // m.def("SupervisedOpfFloatFactory", &SupervisedOpfFloatProxy::OpfFactory, "SupervisedOpfFloatProxy Factory",
    //   "precomputed"_a = false);

    py::class_<SupervisedOpfFloatProxy>(m, "SupervisedOpfFloatProxy")
        .def(py::init())
        .def("fit", &SupervisedOpfFloatProxy::fit)
        .def("predict", &SupervisedOpfFloatProxy::predict)
//        .def("set_distance", (void)&SupervisedOpfFloatProxy::set_distance)
//        .def("set_distance", &SupervisedOpfFloatProxy::set_distance)
        .def_static("SupervisedOpfFloatFactory",  static_cast<SupervisedOpfFloatProxy* (*)(bool , std::string &)>( &OpfFactory), py::return_value_policy::reference)
        .def_static("SupervisedOpfFloatFactory",   static_cast<SupervisedOpfFloatProxy* (*)(bool , py::object &)>( &OpfFactory), py::return_value_policy::reference);
}