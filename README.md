<p align="center">
  <a href="https://github.com/mohammadraziei/libspeech">
    <img src="https://raw.githubusercontent.com/MohammadRaziei/libspeech/master/docs/logo/libspeech-logo-pods.svg" alt="LibSpeech Logo" width="300">
  </a>
  <h2 align="center">LibSpeech</h2>
  <h3 align="center">Lightweight Speech Processing Library</h3>
</p>

<p align="center">
  <em>High-performance C++ library with Python bindings for speech processing powered by ONNX Runtime</em>
</p>

<div align="center">

[![GitHub release](https://img.shields.io/github/release/mohammadraziei/libspeech?color=blue)](https://github.com/mohammadraziei/libspeech/releases)
[![License](https://img.shields.io/badge/license-MIT-purple)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/libspeech)](https://pypi.org/project/libspeech/)
[![C++ Standard](https://img.shields.io/badge/C++-17-blue)](https://en.cppreference.com/w/cpp/17)

[![Build Status](https://github.com/mohammadraziei/libspeech/actions/workflows/build.yml/badge.svg)](https://github.com/mohammadraziei/libspeech/actions)
[![Code Quality](https://sonarcloud.io/api/project_badges/measure?project=MohammadRaziei_libspeech&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=MohammadRaziei_libspeech)
[![CodeFactor](https://www.codefactor.io/repository/github/mohammadraziei/libspeech/badge/master)](https://www.codefactor.io/repository/github/mohammadraziei/libspeech/overview/master)
[![snyk.io](https://snyk.io/advisor/python/libspeech/badge.svg)](https://snyk.io/advisor/python/libspeech)
[![PyPI Downloads](https://img.shields.io/pypi/dm/libspeech)](https://pypi.org/project/libspeech/)


</div>

## ✨ Key Features

- 🚀 **ONNX Runtime Powered**: Efficient AI inference without heavy LibTorch dependency
- 🎙️ **Advanced Audio Processing**: Multi-format support with ultra-low latency
- 🔍 **Feature Extraction**: MFCC, Spectrogram, and 20+ audio features
- 🔊 **Noise Reduction**: AI-powered background noise suppression
- 🖥️ **Cross-Platform**: Windows, Linux & macOS support
- 🐍 **Python Bindings**: Seamless integration with Python applications

## 📦 Installation

### Python Package
```bash
pip install libspeech
```

### From Source
```bash
git clone https://github.com/MohammadRaziei/libspeech.git
cd libspeech
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make install
```

## 🚀 Quick Start

```python
import libspeech

# Initialize audio processor
processor = libspeech.AudioProcessor()

# Load audio file
audio = processor.load_audio("sample.wav")

# Extract features
features = processor.extract_features(audio)

# Apply noise reduction
clean_audio = processor.remove_noise(audio)
```

## 📚 Documentation

- [C++ API Documentation](docs/cpp/README.md)
- [Python API Reference](docs/python/README.md)
- [Examples Directory](examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Sponsors

Developed with ❤️ by the talented engineers at  
[<img src="https://mohammadraziei.github.io/libspeech/images/arman-logo.svg" alt="Arman Tech Hub" width="100">](http://armantechhub.ir/)

[![Stars](https://starchart.cc/mohammadraziei/libspeech.svg?variant=adaptive)](https://starchart.cc/mohammadraziei/libspeech)


