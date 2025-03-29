//
// Created by Mohammad on 3/16/2025.
//

#ifndef LIBSPEECH_PROGRESSBAR_H
#define LIBSPEECH_PROGRESSBAR_H

#include <indicators/progress_bar.hpp>
#include <memory>

namespace speech::utils {
std::shared_ptr<indicators::ProgressBar> createProgressBar(
    const std::string& text = "Processing ",
    indicators::Color color = indicators::Color::green);
}
#endif  // LIBSPEECH_PROGRESSBAR_H
