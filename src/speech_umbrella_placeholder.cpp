// The `speech` target is a pure umbrella library: it exists to bundle
// speech::io + speech::dsp + speech::models into one linkable/installable
// .so for consumers (Python bindings, downstream projects), with no code
// of its own. CMake requires at least one source file for a SHARED library
// target, though, so this file exists purely to satisfy that -- there is
// intentionally nothing to put in it.
