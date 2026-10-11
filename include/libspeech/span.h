//
// span<T> and frame_view<T>: non-owning views over contiguous memory (C++17 stand-ins for
// std::span, plus a "cut into frames" view that the DSP code needs all the time).
//
// Why: the raw-buffer APIs take a (pointer, length) pair or a (pointer, frames, bins) triple that
// the caller has to keep consistent by hand. These types bundle them, convert from std::vector
// automatically, and let you walk a spectrogram or a signal frame by frame:
//
//     for (speech::span<const float> frame : spectrogram.real_frames()) { ... }
//
// Complex data comes in two memory layouts and both are handled without copying or allocating:
//   * interleaved (re, im, re, im, ... -- what NumPy's complex64 and std::complex<float>[] use):
//     a span<std::complex<float>> over the same bytes, see as_complex()/as_floats(); iterating it
//     yields real std::complex<float>& references (a reinterpret_cast, no work at all);
//   * split planes (all re in one array, all im in another -- the layout of ComplexSpectrogram): a
//     reinterpret_cast is impossible there, so split_complex_span iterates with a tiny proxy,
//     complex_ref, that holds the two element references and converts to/from std::complex<float>.
//
// Views never own or free anything; the memory must outlive them.
//
#ifndef LIBSPEECH_SPAN_H
#define LIBSPEECH_SPAN_H

#include <complex>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace speech {

/** A pointer and a length: a view of `size()` consecutive T. Use span<const T> for read-only access. */
template <typename T>
class span {
   public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using pointer = T*;
    using reference = T&;
    using iterator = T*;

    constexpr span() noexcept = default;
    constexpr span(T* data, size_type size) noexcept : data_(data), size_(size) {}

    template <std::size_t N>
    constexpr span(T (&array)[N]) noexcept : data_(array), size_(N) {}

    // From a std::vector (any allocator). A span<const T> also accepts a const vector.
    template <typename U, typename A, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    span(std::vector<U, A>& v) noexcept : data_(v.data()), size_(v.size()) {}
    template <typename U, typename A, typename = std::enable_if_t<std::is_convertible_v<const U*, T*>>>
    span(const std::vector<U, A>& v) noexcept : data_(v.data()), size_(v.size()) {}

    // span<float> -> span<const float>.
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*> && !std::is_same_v<U, T>>>
    constexpr span(const span<U>& other) noexcept : data_(other.data()), size_(other.size()) {}

    [[nodiscard]] constexpr T* data() const noexcept { return data_; }
    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr T& operator[](size_type i) const noexcept { return data_[i]; }
    constexpr T& front() const noexcept { return data_[0]; }
    constexpr T& back() const noexcept { return data_[size_ - 1]; }
    constexpr iterator begin() const noexcept { return data_; }
    constexpr iterator end() const noexcept { return data_ + size_; }

    /** Elements [offset, offset + count); count defaults to "to the end". @throws std::out_of_range */
    [[nodiscard]] span subspan(size_type offset, size_type count = static_cast<size_type>(-1)) const {
        if (offset > size_) throw std::out_of_range("span::subspan: offset is past the end");
        const size_type rest = size_ - offset;
        if (count == static_cast<size_type>(-1)) count = rest;
        if (count > rest) throw std::out_of_range("span::subspan: count is past the end");
        return span(data_ + offset, count);
    }

   private:
    T* data_ = nullptr;
    size_type size_ = 0;
};

/**
 * A buffer seen as `size()` frames of `frame_length()` samples, frame i starting at i * hop().
 *   hop == frame_length : contiguous rows (a (frames x frame_length) matrix, e.g. a spectrogram plane);
 *   hop <  frame_length : overlapping windows over a signal, exactly what STFT frames.
 * Indexing and iteration yield a span<T> per frame, so callers never do the stride arithmetic.
 */
template <typename T>
class frame_view {
   public:
    using size_type = std::size_t;

    class iterator {
       public:
        // The "reference" is a span prvalue, so strictly this is an input iterator; the random-access
        // operations below are all provided and behave as expected for range-for and indexing.
        using iterator_category = std::input_iterator_tag;
        using value_type = span<T>;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = span<T>;

        iterator() = default;
        reference operator*() const { return span<T>(base_ + index_ * hop_, length_); }
        reference operator[](difference_type n) const { return *(*this + n); }

        iterator& operator++() { ++index_; return *this; }
        iterator operator++(int) { iterator t = *this; ++index_; return t; }
        iterator& operator--() { --index_; return *this; }
        iterator operator--(int) { iterator t = *this; --index_; return t; }
        iterator& operator+=(difference_type n) { index_ += static_cast<size_type>(n); return *this; }
        iterator& operator-=(difference_type n) { index_ -= static_cast<size_type>(n); return *this; }
        friend iterator operator+(iterator it, difference_type n) { return it += n; }
        friend iterator operator+(difference_type n, iterator it) { return it += n; }
        friend iterator operator-(iterator it, difference_type n) { return it -= n; }
        friend difference_type operator-(const iterator& a, const iterator& b) {
            return static_cast<difference_type>(a.index_) - static_cast<difference_type>(b.index_);
        }
        friend bool operator==(const iterator& a, const iterator& b) { return a.index_ == b.index_; }
        friend bool operator!=(const iterator& a, const iterator& b) { return a.index_ != b.index_; }
        friend bool operator<(const iterator& a, const iterator& b) { return a.index_ < b.index_; }
        friend bool operator>(const iterator& a, const iterator& b) { return a.index_ > b.index_; }
        friend bool operator<=(const iterator& a, const iterator& b) { return a.index_ <= b.index_; }
        friend bool operator>=(const iterator& a, const iterator& b) { return a.index_ >= b.index_; }

       private:
        friend class frame_view;
        iterator(T* base, size_type index, size_type length, size_type hop)
            : base_(base), index_(index), length_(length), hop_(hop) {}
        T* base_ = nullptr;
        size_type index_ = 0;
        size_type length_ = 0;
        size_type hop_ = 0;
    };

    frame_view() noexcept = default;

    /** Contiguous rows: num_frames * frame_length samples starting at `data`. */
    frame_view(T* data, size_type num_frames, size_type frame_length)
        : frame_view(data, num_frames, frame_length, frame_length) {}

    /** General form; frame i is data[i * hop, i * hop + frame_length). @throws std::invalid_argument */
    frame_view(T* data, size_type num_frames, size_type frame_length, size_type hop)
        : data_(data), num_frames_(num_frames), frame_length_(frame_length), hop_(hop) {
        if (num_frames_ > 0 && (frame_length_ == 0 || hop_ == 0 || data_ == nullptr)) {
            throw std::invalid_argument("frame_view: frames need a non-null buffer, a frame length > 0 and a hop > 0");
        }
    }

    // frame_view<float> -> frame_view<const float>.
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*> && !std::is_same_v<U, T>>>
    frame_view(const frame_view<U>& other) noexcept
        : data_(other.data()), num_frames_(other.size()), frame_length_(other.frame_length()), hop_(other.hop()) {}

    /** `flat` as rows of frame_length samples. @throws std::invalid_argument if it doesn't divide evenly. */
    static frame_view rows(span<T> flat, size_type frame_length) {
        if (frame_length == 0 || flat.size() % frame_length != 0) {
            throw std::invalid_argument("frame_view::rows: the buffer is not a whole number of frames");
        }
        return frame_view(flat.data(), flat.size() / frame_length, frame_length, frame_length);
    }

    /** Overlapping frames over a signal, the way STFT frames it: no padding, so a signal shorter
     *  than one frame has zero frames. @throws std::invalid_argument for a zero length or hop. */
    static frame_view sliding(span<T> signal, size_type frame_length, size_type hop) {
        if (frame_length == 0 || hop == 0) {
            throw std::invalid_argument("frame_view::sliding: frame length and hop must be > 0");
        }
        const size_type frames = signal.size() < frame_length ? 0 : (signal.size() - frame_length) / hop + 1;
        return frame_view(signal.data(), frames, frame_length, hop);
    }

    [[nodiscard]] size_type size() const noexcept { return num_frames_; }   // number of frames
    [[nodiscard]] bool empty() const noexcept { return num_frames_ == 0; }
    [[nodiscard]] size_type frame_length() const noexcept { return frame_length_; }
    [[nodiscard]] size_type hop() const noexcept { return hop_; }
    [[nodiscard]] T* data() const noexcept { return data_; }               // start of frame 0
    [[nodiscard]] bool contiguous() const noexcept { return hop_ == frame_length_; }
    /** Samples from the start of frame 0 to the end of the last frame. */
    [[nodiscard]] size_type extent() const noexcept {
        return num_frames_ == 0 ? 0 : (num_frames_ - 1) * hop_ + frame_length_;
    }

    span<T> operator[](size_type i) const noexcept { return span<T>(data_ + i * hop_, frame_length_); }
    span<T> front() const noexcept { return (*this)[0]; }
    span<T> back() const noexcept { return (*this)[num_frames_ - 1]; }
    iterator begin() const noexcept { return iterator(data_, 0, frame_length_, hop_); }
    iterator end() const noexcept { return iterator(data_, num_frames_, frame_length_, hop_); }

    /** Frames [first, first + count). @throws std::out_of_range */
    [[nodiscard]] frame_view slice(size_type first, size_type count) const {
        if (first > num_frames_ || count > num_frames_ - first) {
            throw std::out_of_range("frame_view::slice: range is past the last frame");
        }
        return frame_view(count == 0 ? data_ : data_ + first * hop_, count, frame_length_, hop_);
    }

    /** All samples of the frames as one span; only meaningful (and only allowed) when contiguous. */
    [[nodiscard]] span<T> flat() const {
        if (!contiguous()) throw std::logic_error("frame_view::flat: overlapping or strided frames are not one block");
        return span<T>(data_, num_frames_ * frame_length_);
    }

   private:
    T* data_ = nullptr;
    size_type num_frames_ = 0;
    size_type frame_length_ = 0;
    size_type hop_ = 0;
};

// ---- Complex data -----------------------------------------------------------------------------

static_assert(sizeof(std::complex<float>) == 2 * sizeof(float) &&
                  alignof(std::complex<float>) == alignof(float),
              "std::complex<float> must be two consecutive floats (array-compatible)");

/**
 * Interleaved floats (re, im, re, im, ...) seen as std::complex<float>, no copy.
 * std::complex<T> is specified to be layout-compatible with T[2], which is what makes this
 * sound. @throws std::invalid_argument if the number of floats is odd.
 */
inline span<std::complex<float>> as_complex(span<float> floats) {
    if (floats.size() % 2 != 0) throw std::invalid_argument("as_complex: an odd number of floats");
    return span<std::complex<float>>(reinterpret_cast<std::complex<float>*>(floats.data()), floats.size() / 2);
}
inline span<const std::complex<float>> as_complex(span<const float> floats) {
    if (floats.size() % 2 != 0) throw std::invalid_argument("as_complex: an odd number of floats");
    return span<const std::complex<float>>(reinterpret_cast<const std::complex<float>*>(floats.data()),
                                           floats.size() / 2);
}

/** The reverse: std::complex<float> values as 2 * size() interleaved floats, no copy. */
inline span<float> as_floats(span<std::complex<float>> values) noexcept {
    return span<float>(reinterpret_cast<float*>(values.data()), values.size() * 2);
}
inline span<const float> as_floats(span<const std::complex<float>> values) noexcept {
    return span<const float>(reinterpret_cast<const float*>(values.data()), values.size() * 2);
}

/**
 * One element of split-plane complex data: references to the real and the imaginary float.
 * Converts to std::complex<float>; assigning a std::complex<float> writes both planes.
 * This is what a split_complex_span iterator hands out -- two references, nothing allocated.
 */
template <typename T>
class complex_ref {
   public:
    using value_type = std::complex<std::remove_cv_t<T>>;

    complex_ref(T& re, T& im) noexcept : re_(re), im_(im) {}
    complex_ref(const complex_ref&) noexcept = default;

    // Value semantics for assignment (a proxy reference): `a[i] = a[j]` copies the number. This
    // must be a plain (non-template) member to be the copy-assignment operator; the reference
    // members would otherwise make it implicitly deleted. Read-only refs cannot be assigned to.
    complex_ref& operator=(const complex_ref& other) noexcept {
        static_assert(!std::is_const_v<T>, "cannot assign through a read-only complex_ref");
        re_ = other.re_;
        im_ = other.im_;
        return *this;
    }
    template <typename U = T, typename = std::enable_if_t<!std::is_const_v<U>>>
    complex_ref& operator=(const value_type& z) noexcept {
        re_ = z.real();
        im_ = z.imag();
        return *this;
    }

    operator value_type() const noexcept { return value_type(re_, im_); }  // NOLINT: implicit on purpose
    [[nodiscard]] value_type value() const noexcept { return value_type(re_, im_); }
    [[nodiscard]] T& real() const noexcept { return re_; }
    [[nodiscard]] T& imag() const noexcept { return im_; }

   private:
    T& re_;
    T& im_;
};

/** Complex values stored as two parallel arrays (split planes): the layout of ComplexSpectrogram. */
template <typename T>
class split_complex_span {
   public:
    using size_type = std::size_t;
    using value_type = std::complex<std::remove_cv_t<T>>;
    using reference = complex_ref<T>;

    class iterator {
       public:
        using iterator_category = std::input_iterator_tag;  // proxy reference; see frame_view::iterator
        using value_type = std::complex<std::remove_cv_t<T>>;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = complex_ref<T>;

        iterator() = default;
        reference operator*() const { return complex_ref<T>(re_[i_], im_[i_]); }
        reference operator[](difference_type n) const { return complex_ref<T>(re_[i_ + n], im_[i_ + n]); }
        iterator& operator++() { ++i_; return *this; }
        iterator operator++(int) { iterator t = *this; ++i_; return t; }
        iterator& operator--() { --i_; return *this; }
        iterator operator--(int) { iterator t = *this; --i_; return t; }
        iterator& operator+=(difference_type n) { i_ += static_cast<size_type>(n); return *this; }
        iterator& operator-=(difference_type n) { i_ -= static_cast<size_type>(n); return *this; }
        friend iterator operator+(iterator it, difference_type n) { return it += n; }
        friend iterator operator+(difference_type n, iterator it) { return it += n; }
        friend iterator operator-(iterator it, difference_type n) { return it -= n; }
        friend difference_type operator-(const iterator& a, const iterator& b) {
            return static_cast<difference_type>(a.i_) - static_cast<difference_type>(b.i_);
        }
        friend bool operator==(const iterator& a, const iterator& b) { return a.i_ == b.i_; }
        friend bool operator!=(const iterator& a, const iterator& b) { return a.i_ != b.i_; }
        friend bool operator<(const iterator& a, const iterator& b) { return a.i_ < b.i_; }

       private:
        friend class split_complex_span;
        iterator(T* re, T* im, size_type i) : re_(re), im_(im), i_(i) {}
        T* re_ = nullptr;
        T* im_ = nullptr;
        size_type i_ = 0;
    };

    split_complex_span() noexcept = default;
    split_complex_span(T* re, T* im, size_type size) noexcept : re_(re), im_(im), size_(size) {}
    /** @throws std::invalid_argument if the planes differ in length. */
    split_complex_span(span<T> re, span<T> im) : re_(re.data()), im_(im.data()), size_(re.size()) {
        if (re.size() != im.size()) throw std::invalid_argument("split_complex_span: real and imag lengths differ");
    }
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*> && !std::is_same_v<U, T>>>
    split_complex_span(const split_complex_span<U>& other) noexcept
        : re_(other.real().data()), im_(other.imag().data()), size_(other.size()) {}

    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] span<T> real() const noexcept { return span<T>(re_, size_); }
    [[nodiscard]] span<T> imag() const noexcept { return span<T>(im_, size_); }

    reference operator[](size_type i) const noexcept { return complex_ref<T>(re_[i], im_[i]); }
    iterator begin() const noexcept { return iterator(re_, im_, 0); }
    iterator end() const noexcept { return iterator(re_, im_, size_); }

   private:
    T* re_ = nullptr;
    T* im_ = nullptr;
    size_type size_ = 0;
};

/**
 * Split-plane complex data cut into frames: frame i is a split_complex_span over row i of the two
 * planes. This is what a ComplexSpectrogram looks like frame by frame:
 *     for (auto frame : spec.frames())          // one spectrum
 *         for (std::complex<float> bin : frame) // one bin, no allocation
 *             energy += std::norm(bin);
 */
template <typename T>
class split_complex_frames {
   public:
    using size_type = std::size_t;

    class iterator {
       public:
        using iterator_category = std::input_iterator_tag;
        using value_type = split_complex_span<T>;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = split_complex_span<T>;

        iterator() = default;
        reference operator*() const { return (*owner_)[i_]; }
        reference operator[](difference_type n) const { return (*owner_)[i_ + static_cast<size_type>(n)]; }
        iterator& operator++() { ++i_; return *this; }
        iterator operator++(int) { iterator t = *this; ++i_; return t; }
        iterator& operator--() { --i_; return *this; }
        iterator& operator+=(difference_type n) { i_ += static_cast<size_type>(n); return *this; }
        friend iterator operator+(iterator it, difference_type n) { return it += n; }
        friend difference_type operator-(const iterator& a, const iterator& b) {
            return static_cast<difference_type>(a.i_) - static_cast<difference_type>(b.i_);
        }
        friend bool operator==(const iterator& a, const iterator& b) { return a.i_ == b.i_; }
        friend bool operator!=(const iterator& a, const iterator& b) { return a.i_ != b.i_; }
        friend bool operator<(const iterator& a, const iterator& b) { return a.i_ < b.i_; }

       private:
        friend class split_complex_frames;
        iterator(const split_complex_frames* owner, size_type i) : owner_(owner), i_(i) {}
        const split_complex_frames* owner_ = nullptr;
        size_type i_ = 0;
    };

    split_complex_frames() noexcept = default;
    /** @throws std::invalid_argument unless both planes have the same frame count, length and hop. */
    split_complex_frames(frame_view<T> re, frame_view<T> im) : re_(re), im_(im) {
        if (re.size() != im.size() || re.frame_length() != im.frame_length() || re.hop() != im.hop()) {
            throw std::invalid_argument("split_complex_frames: real and imag views differ in shape");
        }
    }
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*> && !std::is_same_v<U, T>>>
    split_complex_frames(const split_complex_frames<U>& other) noexcept : re_(other.real()), im_(other.imag()) {}

    [[nodiscard]] size_type size() const noexcept { return re_.size(); }  // number of frames
    [[nodiscard]] bool empty() const noexcept { return re_.empty(); }
    [[nodiscard]] size_type frame_length() const noexcept { return re_.frame_length(); }
    [[nodiscard]] frame_view<T> real() const noexcept { return re_; }
    [[nodiscard]] frame_view<T> imag() const noexcept { return im_; }

    split_complex_span<T> operator[](size_type i) const noexcept {
        return split_complex_span<T>(re_[i], im_[i]);
    }
    iterator begin() const noexcept { return iterator(this, 0); }
    iterator end() const noexcept { return iterator(this, size()); }

   private:
    frame_view<T> re_;
    frame_view<T> im_;
};

}  // namespace speech

#endif  // LIBSPEECH_SPAN_H
