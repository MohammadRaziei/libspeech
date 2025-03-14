#include "audio.h"

int main() {
    Audio audio;

    // Load an audio file
    if (!audio.load("D:\\Desk\\Work\\Academy-hamrah\\courses\\assets\\en.wav")) {
        return -1;
    }

    // Play the audio
    audio.play();

    // Save the audio to a new file
//    if (!audio.save("D:\\Desk\\Work\\Academy-hamrah\\courses\\assets\\output.wav")) {
//        return -1;
//    }

    return 0;
}