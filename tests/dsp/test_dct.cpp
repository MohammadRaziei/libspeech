#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/dct.h"

UTEST(DctII, OutputLengthMatchesInputByDefault) {
    std::vector<float> input(40, 1.0f);
    auto out = speech::dsp::dctII(input);
    ASSERT_EQ(out.size(), static_cast<size_t>(40));
}

UTEST(DctII, TruncatesToRequestedNumOutputs) {
    std::vector<float> input(40, 1.0f);
    auto out = speech::dsp::dctII(input, 13);
    ASSERT_EQ(out.size(), static_cast<size_t>(13));
}

UTEST(DctII, ConstantInputHasEnergyOnlyInDcTerm) {
    // Mathematical property of DCT-II: for a constant input, every AC
    // coefficient (k>0) is exactly zero (the cosine basis functions are
    // orthogonal to the constant function for k>0 over this summation).
    // Only the DC term (k=0) should be non-zero.
    std::vector<float> input(32, 3.0f);
    auto out = speech::dsp::dctII(input);

    ASSERT_TRUE(std::fabs(out[0]) > 1.0f);
    for (size_t k = 1; k < out.size(); ++k) {
        ASSERT_TRUE(std::fabs(out[k]) < 1e-3f);
    }
}

UTEST(DctII, EmptyInputThrows) {
    std::vector<float> empty;
    bool threw = false;
    try {
        speech::dsp::dctII(empty);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(DctII, NumOutputsOutOfRangeThrows) {
    std::vector<float> input(10, 1.0f);
    bool threw = false;
    try {
        speech::dsp::dctII(input, 11);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST_MAIN();
