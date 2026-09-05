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
        add_python_test(dsp)
    endif()
endif()
