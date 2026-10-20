#include "utest.h"

#include <cmath>
#include <limits>
#include <vector>

#include "level_match.h"

using speech::models::detail::match_level;

namespace {
double rms(const std::vector<float>& v) {
    double e = 0.0;
    for (const float s : v) e += static_cast<double>(s) * s;
    return std::sqrt(e / static_cast<double>(v.size()));
}
double peak(const std::vector<float>& v) {
    double p = 0.0;
    for (const float s : v) p = std::fmax(p, std::fabs(static_cast<double>(s)));
    return p;
}
std::vector<float> sine(std::size_t n, float amplitude, double cycles) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = amplitude * static_cast<float>(std::sin(6.283185307179586 * cycles * static_cast<double>(i) / static_cast<double>(n)));
    return v;
}
}  // namespace

UTEST(level_match, a_far_too_loud_signal_is_brought_back_to_the_reference_rms) {
    const std::vector<float> reference = sine(16000, 0.1f, 50);
    std::vector<float> signal = sine(16000, 3.0f, 50);  // 30x louder, samples far beyond +-1
    match_level(reference.data(), reference.size(), signal);
    ASSERT_NEAR(rms(signal), rms(reference), 1e-4);
    ASSERT_LE(peak(signal), 0.99);
}

UTEST(level_match, a_too_quiet_signal_is_raised) {
    const std::vector<float> reference = sine(16000, 0.2f, 40);
    std::vector<float> signal = sine(16000, 0.001f, 40);
    match_level(reference.data(), reference.size(), signal);
    ASSERT_NEAR(rms(signal), rms(reference), 1e-4);
}

UTEST(level_match, the_peak_ceiling_wins_over_the_rms) {
    // Matching the RMS of this spiky signal to a steady reference would put its spikes far beyond full scale.
    const std::vector<float> reference = sine(16000, 0.7f, 50);  // RMS 0.49, peak 0.7
    std::vector<float> signal(16000, 0.0f);
    for (std::size_t i = 0; i < signal.size(); i += 1000) signal[i] = 10.0f;
    match_level(reference.data(), reference.size(), signal);
    ASSERT_NEAR(peak(signal), 0.99, 1e-5);   // held at the ceiling ...
    ASSERT_LT(rms(signal), rms(reference));  // ... so the RMS is lower than the reference's
}

UTEST(level_match, a_reference_above_full_scale_allows_a_higher_ceiling) {
    const std::vector<float> reference = sine(16000, 2.0f, 30);  // float audio beyond +-1
    std::vector<float> signal = sine(16000, 40.0f, 30);
    match_level(reference.data(), reference.size(), signal);
    ASSERT_NEAR(rms(signal), rms(reference), 1e-3);
    ASSERT_NEAR(peak(signal), 2.0, 1e-3);
}

UTEST(level_match, silence_and_non_finite_values_leave_the_signal_alone) {
    const std::vector<float> loud = sine(1000, 2.0f, 10);
    std::vector<float> silent_reference_target = loud;
    const std::vector<float> silence(1000, 0.0f);
    match_level(silence.data(), silence.size(), silent_reference_target);
    ASSERT_TRUE(silent_reference_target == loud);

    std::vector<float> silent_signal(1000, 0.0f);
    match_level(loud.data(), loud.size(), silent_signal);
    ASSERT_TRUE(silent_signal == silence);

    std::vector<float> with_nan = loud;
    with_nan[5] = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> before = with_nan;
    match_level(loud.data(), loud.size(), with_nan);
    ASSERT_TRUE(std::isnan(with_nan[5]));
    ASSERT_TRUE(with_nan[0] == before[0]);

    std::vector<float> empty;
    match_level(loud.data(), loud.size(), empty);
    ASSERT_TRUE(empty.empty());
    std::vector<float> unchanged = loud;
    match_level(nullptr, 0, unchanged);
    ASSERT_TRUE(unchanged == loud);
}
