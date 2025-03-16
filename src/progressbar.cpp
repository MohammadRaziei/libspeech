//
// Created by Mohammad on 3/16/2025.
//
#include "progressbar.h"


std::shared_ptr<indicators::ProgressBar> speech::bar::createProgressBar(
    const std::string& text, indicators::Color color) {

    // Create a shared_ptr to manage the ProgressBar object
    auto progressBar = std::make_shared<indicators::ProgressBar>(
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::ForegroundColor{color},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::PrefixText{text},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    );

    return progressBar; // Return the shared_ptr
}