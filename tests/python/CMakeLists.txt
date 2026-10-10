# Python tests -- see tests/CMakeLists.txt for the full
# speech_test_<language>_<module> naming scheme this fits into. Split by
# module (audio / models) the same way the C++ side is, even though both
# currently run via one pytest process each on a single file -- this keeps
# the naming symmetric and lets either be run/skipped independently.
#
# Included from CMakeLists.txt only after the Python bindings are actually
# configured (needs the `_audio`/`_models` targets and ${PYTHON_PROJECT_NAME}
# to exist), unlike tests/CMakeLists.txt which is included early enough to
# work standalone under -DBUILD_MODELS=OFF.

if(TARGET speech_io_py)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pytest --version
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE PYTEST_CHECK_RESULT
    )

    if(NOT PYTEST_CHECK_RESULT EQUAL 0)
        message(WARNING "pytest is not installed. Python tests will be skipped. Install it with: pip install pytest")
    else()
        function(add_python_test module)
            add_test(NAME python_${module}
                     COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_${module}.py
                     WORKING_DIRECTORY ${PYTHON_PROJECT_NAME}
            )

            add_custom_target(speech_test_python_${module}
                COMMAND ${CMAKE_COMMAND} -E echo "-- Running Python ${module} tests..."
                COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_${module}.py
                WORKING_DIRECTORY ${PYTHON_PROJECT_NAME}
                USES_TERMINAL
            )

            # PYTHON_BIND_MODULES (set in the main CMakeLists.txt) lists
            # every discovered binding module (_about, _audio, _models,
            # ...) -- depend on all of them, not a hardcoded subset, so a
            # future binding module can never again silently miss being
            # built before these tests run (that's exactly the bug that
            # once caused a real ModuleNotFoundError for _models).
            add_dependencies(speech_test_python_${module} ${PYTHON_BIND_MODULES} speech)
            add_dependencies(speech_test_python speech_test_python_${module})
        endfunction()

        add_python_test(audio)
        add_python_test(models)
        add_python_test(logging)
        add_python_test(dsp)
        add_python_test(numpy_api)
        add_python_test(simd)
        add_python_test(librosa_compat)   # needs the `test` extra (librosa, soundfile, scipy); skips itself without

        # Heavy end-to-end Denoiser test (real ONNX model: download + ~1 GB RAM). It is marked
        # `models`, so every normal pytest run skips it (tests/python/conftest.py) -- the wheel
        # tests in wheels.yml / cibuildwheel, speech_coverage_python, a plain local run. It runs
        # only when asked for, via this target, which cmake.yml builds. Deliberately NOT part of
        # speech_test_python or ctest.
        add_custom_target(speech_test_python_models_e2e
            COMMAND ${CMAKE_COMMAND} -E echo "-- Running Python model tests: pytest -m models"
            COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_models.py -m models
            WORKING_DIRECTORY ${PYTHON_PROJECT_NAME}
            USES_TERMINAL
        )
        add_dependencies(speech_test_python_models_e2e ${PYTHON_BIND_MODULES} speech)

        # ---------------------------------------------------------------
        # Python coverage (coverage.py -> lcov, same flow as ctoon's
        # ctoon_coverage_python). Needs `pip install coverage` and
        # -DBUILD_COVERAGE=ON; covers the Python package's own .py files
        # (the compiled nanobind modules are covered on the C++ side).
        # ---------------------------------------------------------------
        if(SPEECH_COVERAGE_ENABLED)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -m coverage --version
                OUTPUT_QUIET
                ERROR_QUIET
                RESULT_VARIABLE COVERAGE_CHECK_RESULT
            )
            if(COVERAGE_CHECK_RESULT EQUAL 0)
                set(_py_cov_dir ${SPEECH_COVERAGE_BINARY_DIR}/python)
                file(MAKE_DIRECTORY ${_py_cov_dir})

                add_custom_target(speech_coverage_python
                    # Clean previous coverage data (only html, keep index.html wrapper)
                    COMMAND ${CMAKE_COMMAND} -E remove_directory ${_py_cov_dir}/html
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${_py_cov_dir}/html

                    # Run the tests with coverage enabled, from the directory
                    # the package was built into so `import libspeech` finds it.
                    COMMAND ${CMAKE_COMMAND} -E echo "-- Capturing coverage..."
                    COMMAND ${Python3_EXECUTABLE} -m coverage run
                            --source=${PYTHON_PROJECT_NAME}/${PROJECT_NAME}
                            --omit="*/tests/*"
                            -m pytest ${CMAKE_CURRENT_SOURCE_DIR}

                    COMMAND ${CMAKE_COMMAND} -E echo "-- Generating coverage.lcov file..."
                    COMMAND ${Python3_EXECUTABLE} -m coverage lcov -o ${_py_cov_dir}/coverage.lcov
                    # coverage.py writes paths relative to the build dir
                    # (libspeech/__init__.py); map them onto the source tree.
                    COMMAND ${CMAKE_COMMAND}
                            -DCOV_FILE=${_py_cov_dir}/coverage.lcov
                            -DPREFIX_PATH=src/bindings/python/
                            -P ${PROJECT_SOURCE_DIR}/cmake/FixLCovPaths.cmake

                    COMMAND ${CMAKE_COMMAND} -E echo "-- Generating HTML report..."
                    COMMAND ${Python3_EXECUTABLE} -m coverage html -d ${_py_cov_dir}/html
                    COMMAND ${CMAKE_COMMAND} -E echo "-- HTML entry point created here: ${_py_cov_dir}/html/index.html"
                    COMMAND ${CMAKE_COMMAND} -E echo "-- Coverage summary:"
                    COMMAND ${Python3_EXECUTABLE} -m coverage report

                    WORKING_DIRECTORY ${PYTHON_PROJECT_NAME}
                    USES_TERMINAL
                )
                add_dependencies(speech_coverage_python ${PYTHON_BIND_MODULES} speech)
                add_dependencies(speech_coverage speech_coverage_python)

                add_custom_command(TARGET speech_coverage_python POST_BUILD
                    COMMAND ${CMAKE_COMMAND}
                        -DIFRAME_SRC="html/index.html"
                        -DSECTION_TITLE="Python-Bindings"
                        -DLOGO_SVG="${PROJECT_SOURCE_DIR}/docs/logo/libspeech-logo.svg"
                        -DPROJECT_NAME="${PROJECT_NAME}"
                        -DOUTPUT_DIR="${_py_cov_dir}"
                        -P "${PROJECT_SOURCE_DIR}/cmake/GenerateIframe.cmake"
                    COMMENT "Generating iframe wrapper for Python coverage report"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                )
            else()
                message(WARNING "coverage module is not installed. Python coverage target will be skipped. Install it with: pip install coverage")
            endif()
        endif()
    endif()
endif()
