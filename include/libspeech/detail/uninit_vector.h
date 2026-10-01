//
// UninitVector<T>: std::vector whose resize()/size-constructor leaves
// trivially-constructible elements UNINITIALIZED instead of zero-filling them.
//
// Used for output buffers the DSP kernels overwrite in full (spectrograms,
// feature matrices, resampled audio): the zero-fill pass would touch every
// page once just to have the kernel write it again. Do NOT use it for
// buffers a kernel accumulates into (istft's output) -- those need zeros.
//
#ifndef LIBSPEECH_DETAIL_UNINIT_VECTOR_H
#define LIBSPEECH_DETAIL_UNINIT_VECTOR_H

#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace speech::detail {

template <typename T, typename A = std::allocator<T>>
class DefaultInitAllocator : public A {
    using Traits = std::allocator_traits<A>;

   public:
    template <typename U>
    struct rebind {
        using other = DefaultInitAllocator<U, typename Traits::template rebind_alloc<U>>;
    };

    using A::A;

    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(ptr)) U;  // default-init: no zeroing for scalars
    }
    template <typename U, typename... Args>
    void construct(U* ptr, Args&&... args) {
        Traits::construct(static_cast<A&>(*this), ptr, std::forward<Args>(args)...);
    }
};

template <typename T>
using UninitVector = std::vector<T, DefaultInitAllocator<T>>;

}  // namespace speech::detail

#endif  // LIBSPEECH_DETAIL_UNINIT_VECTOR_H
