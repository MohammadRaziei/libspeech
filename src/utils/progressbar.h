//
// Created by Mohammad on 3/16/2025.
//

#ifndef LIBSPEECH_PROGRESSBAR_H
#define LIBSPEECH_PROGRESSBAR_H

#include <httpp/progress.hpp>
#include <memory>
#include <string>

namespace speech::utils {
// Percent-based progress bar (0..100), backed by httpp::progress::bar.
// httpp hides the `indicators` dependency behind its own pointer, so no
// third-party progress-bar library needs to be a submodule/dependency of
// libspeech itself anymore.
std::shared_ptr<httpp::progress::bar> createProgressBar(
    const std::string& text = "Processing ");
}
#endif  // LIBSPEECH_PROGRESSBAR_H
