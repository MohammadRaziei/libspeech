#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/dct.h"

UTEST(dct_ii_plan, output_length_matches_input_by_default) {
    std::vector<float> input(40, 1.0f);
    auto out = speech::dsp::dct_ii(input);
    ASSERT_EQ(out.size(), static_cast<size_t>(40));
}

UTEST(dct_ii_plan, truncates_to_requested_num_outputs) {
    std::vector<float> input(40, 1.0f);
    auto out = speech::dsp::dct_ii(input, 13);
    ASSERT_EQ(out.size(), static_cast<size_t>(13));
}

UTEST(dct_ii_plan, constant_input_has_energy_only_in_dc_term) {
    // Mathematical property of DCT-II: for a constant input, every AC
    // coefficient (k>0) is exactly zero (the cosine basis functions are
    // orthogonal to the constant function for k>0 over this summation).
    // Only the DC term (k=0) should be non-zero.
    std::vector<float> input(32, 3.0f);
    auto out = speech::dsp::dct_ii(input);

    ASSERT_TRUE(std::fabs(out[0]) > 1.0f);
    for (size_t k = 1; k < out.size(); ++k) {
        ASSERT_TRUE(std::fabs(out[k]) < 1e-3f);
    }
}

UTEST(dct_ii_plan, empty_input_throws) {
    std::vector<float> empty;
    bool threw = false;
    try {
        speech::dsp::dct_ii(empty);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(dct_ii_plan, num_outputs_out_of_range_throws) {
    std::vector<float> input(10, 1.0f);
    bool threw = false;
    try {
        speech::dsp::dct_ii(input, 11);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

