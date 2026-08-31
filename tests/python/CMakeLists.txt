# Python tests (pytest), wired the same way ctoon does it (see
# tests/python/CMakeLists.txt in github.com/mohammadraziei/ctoon): a
# `speech_test_python` target that actually runs pytest against the built
# package, added as a dependency of the top-level `speech_test` umbrella
# (created in tests/CMakeLists.txt) -- so `cmake --build . --target
# speech_test` runs DSP *and* Python tests together, not just DSP.
#
# Included from CMakeLists.txt only after the Python bindings are actually
# configured (it needs the `_audio` target and ${PYTHON_PROJECT_NAME} to
# exist), unlike tests/CMakeLists.txt which is included early enough to
# work standalone under -DBUILD_MODELS=OFF.

if(TARGET _audio)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pytest --version
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE PYTEST_CHECK_RESULT
    )

    if(NOT PYTEST_CHECK_RESULT EQUAL 0)
        message(WARNING "pytest is not installed. Python tests will be skipped. Install it with: pip install pytest")
    else()
        add_test(NAME speech_python
                 COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
                 WORKING_DIRECTORY ${PYTHON_PROJECT_NAME}
        )

        add_custom_target(speech_test_python
            COMMAND ${CMAKE_COMMAND} -E echo "-- Running Python tests..."
            COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
            WORKING_DIRECTORY ${PYTHON_PROJECT_NAME}
            USES_TERMINAL
        )

        # _audio/_about/speech all need to be built (and land inside the
        # package directory) before pytest can `import libspeech`.
        add_dependencies(speech_test_python _audio _about speech)
        add_dependencies(speech_test speech_test_python)
    endif()
endif()
