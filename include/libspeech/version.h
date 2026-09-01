//
// Single source of truth for libspeech's version. Read by:
//   - cmake/DynamicVersion.cmake, to set project(... VERSION ...) in CMakeLists.txt
//   - pyproject.toml's [tool.scikit-build.metadata.version] (regex provider)
//   - version.py, the maintainer CLI for showing/bumping/tagging releases
//     (run `python version.py --help`)
//
// Do not hand-edit the individual numbers for a release -- use
// `python version.py tag create [major|minor|patch|X.Y.Z]` instead, which
// keeps this header, git tags, and (transitively) the Python package
// version all in sync.
//

#ifndef LIBSPEECH_VERSION_H
#define LIBSPEECH_VERSION_H

#define LIBSPEECH_VERSION_MAJOR 0
#define LIBSPEECH_VERSION_MINOR 0
#define LIBSPEECH_VERSION_PATCH 1

#define LIBSPEECH_VERSION_STRINGIFY_IMPL(x) #x
#define LIBSPEECH_VERSION_STRINGIFY(x) LIBSPEECH_VERSION_STRINGIFY_IMPL(x)

#define LIBSPEECH_VERSION_STRING                                     \
    LIBSPEECH_VERSION_STRINGIFY(LIBSPEECH_VERSION_MAJOR) "."          \
    LIBSPEECH_VERSION_STRINGIFY(LIBSPEECH_VERSION_MINOR) "."          \
    LIBSPEECH_VERSION_STRINGIFY(LIBSPEECH_VERSION_PATCH)

#endif  // LIBSPEECH_VERSION_H
