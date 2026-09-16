//
// Created by Mohammad on 3/16/2025.
//
#include "utils/progressbar.h"

std::shared_ptr<httpp::progress::bar> speech::utils::createProgressBar(
    const std::string& text) {
    // httpp::progress::bar is percent-based here (total = 100), matching
    // the set_progress(0..100) call sites in this codebase (downloadFile,
    // audio playback simulation, ...).
    return std::make_shared<httpp::progress::bar>(100, text);
}
