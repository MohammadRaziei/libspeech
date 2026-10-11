// Zero-copy helpers shared by the nanobind modules: hand a C++ buffer to
// NumPy without copying it, by moving the buffer onto the heap and letting a
// capsule free it when the last array referencing it dies.
#ifndef LIBSPEECH_BINDINGS_NDARRAY_UTIL_H
#define LIBSPEECH_BINDINGS_NDARRAY_UTIL_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <complex>
#include <cstddef>
#include <memory>
#include <utility>

#include "libspeech/detail/uninit_vector.h"

namespace nb = nanobind;

namespace speech::py {

using float_vec = speech::detail::uninit_vector<float>;
using in_array_1d = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using in_array_2d = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using out_array_1d = nb::ndarray<nb::numpy, float, nb::ndim<1>>;
using out_array_2d = nb::ndarray<nb::numpy, float, nb::ndim<2>>;

// complex64 <-> std::complex<float>: both are two consecutive float32 (re, im), so a NumPy complex64
// array and a std::complex<float> buffer are the same bytes and nothing is converted.
using cx = std::complex<float>;
using cx_vec = speech::detail::uninit_vector<cx>;
using in_cx_array_1d = nb::ndarray<const cx, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using in_cx_array_2d = nb::ndarray<const cx, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using out_cx_array_1d = nb::ndarray<nb::numpy, cx, nb::ndim<1>>;
using out_cx_array_2d = nb::ndarray<nb::numpy, cx, nb::ndim<2>>;

// Takes ownership of `v`; the returned array views its storage (no copy).
template <typename Array>
Array wrap(float_vec&& v, std::initializer_list<std::size_t> shape) {
    auto* owner = new float_vec(std::move(v));
    nb::capsule capsule(owner, [](void* p) noexcept { delete static_cast<float_vec*>(p); });
    return Array(owner->data(), shape, capsule);
}

// Same for complex output.
template <typename Array>
Array wrap_complex(cx_vec&& v, std::initializer_list<std::size_t> shape) {
    auto* owner = new cx_vec(std::move(v));
    nb::capsule capsule(owner, [](void* p) noexcept { delete static_cast<cx_vec*>(p); });
    return Array(owner->data(), shape, capsule);
}

}  // namespace speech::py

#endif  // LIBSPEECH_BINDINGS_NDARRAY_UTIL_H
