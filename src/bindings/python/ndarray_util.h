// Zero-copy helpers shared by the nanobind modules: hand a C++ buffer to
// NumPy without copying it, by moving the buffer onto the heap and letting a
// capsule free it when the last array referencing it dies.
#ifndef LIBSPEECH_BINDINGS_NDARRAY_UTIL_H
#define LIBSPEECH_BINDINGS_NDARRAY_UTIL_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "libspeech/detail/uninit_vector.h"

namespace nb = nanobind;

namespace speech::py {

using FloatVec = speech::detail::UninitVector<float>;
using InArray1D = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using InArray2D = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutArray1D = nb::ndarray<nb::numpy, float, nb::ndim<1>>;
using OutArray2D = nb::ndarray<nb::numpy, float, nb::ndim<2>>;

// Takes ownership of `v`; the returned array views its storage (no copy).
template <typename Array>
Array wrap(FloatVec&& v, std::initializer_list<std::size_t> shape) {
    auto* owner = new FloatVec(std::move(v));
    nb::capsule capsule(owner, [](void* p) noexcept { delete static_cast<FloatVec*>(p); });
    return Array(owner->data(), shape, capsule);
}

}  // namespace speech::py

#endif  // LIBSPEECH_BINDINGS_NDARRAY_UTIL_H
