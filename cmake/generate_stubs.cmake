# Writes a .pyi stub for each compiled nanobind module, the way ctoon does (python -m
# nanobind.stubgen), but through the package: libspeech's modules need the shared libraries that
# libspeech/__init__.py preloads, so they can only be imported as `libspeech.<module>` from an
# assembled copy of the package (the build tree, or the install tree a wheel is made from).
#
# Run with: cmake -DPYTHON_EXECUTABLE=... -DPACKAGE_ROOT=<dir holding libspeech/> -DPACKAGE_NAME=libspeech
#                 -DMODULES="speech_dsp_py|speech_io_py|..." -P generate_stubs.cmake
# A module that cannot be imported (e.g. a cross-compiled build) only costs its stub: this warns and
# carries on, it never fails the build.
string(REPLACE "|" ";" _modules "${MODULES}")
foreach(_module IN LISTS _modules)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -m nanobind.stubgen
                -m "${PACKAGE_NAME}.${_module}"
                -o "${PACKAGE_ROOT}/${PACKAGE_NAME}/${_module}.pyi"
                -i "${PACKAGE_ROOT}"
        RESULT_VARIABLE _rc
        OUTPUT_QUIET
        ERROR_VARIABLE _err)
    if(_rc)
        message(WARNING "libspeech: could not generate the stub for ${_module} (exit ${_rc}); skipping it.\n${_err}")
    else()
        message(STATUS "libspeech: wrote ${PACKAGE_NAME}/${_module}.pyi")
    endif()
endforeach()
