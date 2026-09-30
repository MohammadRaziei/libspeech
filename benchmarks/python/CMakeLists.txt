set(PY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/venv")
set(RESULTS_DIR "${CMAKE_CURRENT_BINARY_DIR}/results")
set(SCRATCH_DIR "${CMAKE_CURRENT_BINARY_DIR}/scratch")

if(WIN32)
  set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
else()
  set(VENV_PYTHON "${VENV_DIR}/bin/python")
endif()
# Every venv-python invocation below passes -W ignore::RuntimeWarning:site:
# Ninja relativizes absolute in-build-tree paths ("report/../python/venv/bin/
# python"), which is functionally identical but trips a sys.prefix/pyvenv.cfg
# consistency check Python 3.11+ added to the frozen `site` module. Scoped to
# that one module so it never hides a real RuntimeWarning from anything else.

file(MAKE_DIRECTORY "${RESULTS_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

# ---------------------------------------------------------------- venv --
# Runs at CONFIGURE time, so `cmake -S . -B build` alone leaves a ready venv
# and `cmake --build` only runs benchmarks. libspeech itself is installed
# from a fresh git checkout (LIBSPEECH_BENCH_GITHUB_URL) exactly like the
# competitors come from PyPI -- never from the local source tree.
#
# Guarded by a stamp file: reconfiguring does NOT reinstall unless the venv is
# missing. Delete the stamp (or the build directory) after changing
# requirements.txt or LIBSPEECH_BENCH_GITHUB_URL.

set(VENV_STAMP "${CMAKE_CURRENT_BINARY_DIR}/.venv_ready")

if(NOT EXISTS "${VENV_STAMP}")
  message(STATUS "libspeech benchmarks: creating venv at ${VENV_DIR} (configure time)...")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m venv "${VENV_DIR}"
    RESULT_VARIABLE _ls_venv_rc
  )
  if(NOT _ls_venv_rc EQUAL 0)
    message(FATAL_ERROR "libspeech benchmarks: failed to create venv at ${VENV_DIR}")
  endif()

  message(STATUS "libspeech benchmarks: installing competitors + libspeech from ${LIBSPEECH_BENCH_GITHUB_URL} (configure time)...")
  execute_process(
    COMMAND "${VENV_PYTHON}" -W "ignore::RuntimeWarning:site" -m pip install --upgrade pip --quiet
    RESULT_VARIABLE _ls_pip_upgrade_rc
  )
  if(NOT _ls_pip_upgrade_rc EQUAL 0)
    message(FATAL_ERROR "libspeech benchmarks: failed to upgrade pip in ${VENV_DIR}")
  endif()

  execute_process(
    COMMAND "${VENV_PYTHON}" -W "ignore::RuntimeWarning:site" -m pip install -r "${PY_DIR}/requirements.txt" --quiet
    RESULT_VARIABLE _ls_pip_req_rc
  )
  if(NOT _ls_pip_req_rc EQUAL 0)
    message(FATAL_ERROR "libspeech benchmarks: failed to install ${PY_DIR}/requirements.txt")
  endif()

  # Building libspeech from source compiles the C++ core and downloads
  # ONNX Runtime, so this step is the slow one (minutes on a cold cache).
  # No --no-deps here (unlike a pure-Python package): libspeech has one real
  # runtime dependency, `httpp`, and it fails to import without it.
  execute_process(
    COMMAND "${VENV_PYTHON}" -W "ignore::RuntimeWarning:site" -m pip install --force-reinstall "${LIBSPEECH_BENCH_GITHUB_URL}" --quiet
    RESULT_VARIABLE _ls_pip_pkg_rc
  )
  if(NOT _ls_pip_pkg_rc EQUAL 0)
    message(FATAL_ERROR "libspeech benchmarks: failed to install libspeech from ${LIBSPEECH_BENCH_GITHUB_URL}")
  endif()

  file(TOUCH "${VENV_STAMP}")
else()
  message(STATUS "libspeech benchmarks: venv already present at ${VENV_DIR} -- skipping "
                  "(delete ${VENV_STAMP} to force recreation)")
endif()

add_custom_target(libspeech_bench_venv)

# -------------------------------------------------------------- corpus --

set(CORPUS_DIR "${SCRATCH_DIR}/corpus")
set(CORPUS_MANIFEST "${CORPUS_DIR}/manifest.json")

add_custom_command(
  OUTPUT "${CORPUS_MANIFEST}"
  COMMAND "${VENV_PYTHON}" -W "ignore::RuntimeWarning:site" "${PY_DIR}/corpus.py" "${CORPUS_DIR}"
  DEPENDS "${VENV_STAMP}" "${PY_DIR}/corpus.py"
  COMMENT "libspeech benchmarks: generating the synthetic audio corpus"
  USES_TERMINAL
)
add_custom_target(libspeech_bench_corpus DEPENDS "${CORPUS_MANIFEST}")

# --------------------------------------------------------------- sizes --

set(SIZES_JSON "${RESULTS_DIR}/sizes.json")

add_custom_command(
  OUTPUT "${SIZES_JSON}"
  COMMAND "${VENV_PYTHON}" -W "ignore::RuntimeWarning:site" "${PY_DIR}/bench_sizes.py" "${SIZES_JSON}"
  DEPENDS "${VENV_STAMP}" "${PY_DIR}/bench_sizes.py"
  COMMENT "libspeech benchmarks: comparing install footprint (pip-size)"
  USES_TERMINAL
)
add_custom_target(libspeech_bench_sizes DEPENDS "${SIZES_JSON}")

# ---------------------------------------------------------- throughput --
# Repeats per (operation, library, corpus entry) cell; the best (min) is kept.
# Configurable because "how many repeats" is a real noise-vs-CI-time tradeoff.

set(LIBSPEECH_BENCH_REPEATS "7" CACHE STRING
    "Timed repeats per (operation, library, corpus entry) cell in bench_throughput.py; best (min) is kept")

set(THROUGHPUT_JSON "${RESULTS_DIR}/throughput.json")

add_custom_command(
  OUTPUT "${THROUGHPUT_JSON}"
  COMMAND "${VENV_PYTHON}" -W "ignore::RuntimeWarning:site" "${PY_DIR}/bench_throughput.py"
          "${CORPUS_MANIFEST}" "${THROUGHPUT_JSON}" --repeats "${LIBSPEECH_BENCH_REPEATS}"
  DEPENDS "${VENV_STAMP}" "${PY_DIR}/bench_throughput.py" "${PY_DIR}/ops.py" "${CORPUS_MANIFEST}"
  COMMENT "libspeech benchmarks: load / resample / stft / mfcc throughput (${LIBSPEECH_BENCH_REPEATS} repeats/cell)"
  USES_TERMINAL
)
add_custom_target(libspeech_bench_throughput DEPENDS "${THROUGHPUT_JSON}")

# -------------------------------------------------------------- memory --
# One driver, one fresh child process per data point. The driver never loads
# audio or imports a benchmarked library itself, so no measurement inherits
# its parent's peak-RSS history (see bench_op_memory_one.py).

set(MEMORY_JSON "${RESULTS_DIR}/memory.json")

add_custom_command(
  OUTPUT "${MEMORY_JSON}"
  COMMAND "${VENV_PYTHON}" -W "ignore::RuntimeWarning:site" "${PY_DIR}/bench_op_memory_driver.py"
          "${CORPUS_MANIFEST}" "${MEMORY_JSON}"
  DEPENDS "${VENV_STAMP}" "${PY_DIR}/bench_op_memory_driver.py" "${PY_DIR}/bench_op_memory_one.py"
          "${PY_DIR}/ops.py" "${CORPUS_MANIFEST}"
  COMMENT "libspeech benchmarks: peak memory for load / resample / stft / mfcc"
  USES_TERMINAL
)
add_custom_target(libspeech_bench_memory DEPENDS "${MEMORY_JSON}")

# --------------------------------------------------------- system info --
# What machine this ran on, reported plainly in the report footer.

set(SYSTEM_INFO_JSON "${RESULTS_DIR}/system_info.json")

add_custom_command(
  OUTPUT "${SYSTEM_INFO_JSON}"
  COMMAND "${VENV_PYTHON}" -W "ignore::RuntimeWarning:site" "${PY_DIR}/system_info.py" "${SYSTEM_INFO_JSON}"
  DEPENDS "${VENV_STAMP}" "${PY_DIR}/system_info.py"
  COMMENT "libspeech benchmarks: recording machine specs (CPU/RAM/OS)"
  USES_TERMINAL
)
add_custom_target(libspeech_bench_system_info DEPENDS "${SYSTEM_INFO_JSON}")

# Expose to the parent scope (report/CMakeLists.txt needs these)
set(LIBSPEECH_BENCH_RESULTS_SCRATCH_DIR "${RESULTS_DIR}" PARENT_SCOPE)
set(LIBSPEECH_BENCH_VENV_PYTHON "${VENV_PYTHON}" PARENT_SCOPE)
set(LIBSPEECH_BENCH_VENV_STAMP "${VENV_STAMP}" PARENT_SCOPE)
set(LIBSPEECH_BENCH_ALL_RESULT_JSONS
    "${THROUGHPUT_JSON}" "${MEMORY_JSON}" "${SIZES_JSON}" "${SYSTEM_INFO_JSON}"
    PARENT_SCOPE)
