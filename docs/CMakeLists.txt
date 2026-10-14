# libspeech documentation build, laid out like ctoon's docs/CMakeLists.txt:
#   C++    -> Doxygen   (docs/cpp/CreateDocs.cmake)
#   Python -> Sphinx    (docs/python/CreateDocs.cmake, imports the staged package of the build tree)
#   index  -> docs/generate_index.py (supported_langs.json + per-language fragments)
#   pages  -> Markdown pages (SpeechPages.cmake): changelog, API renames, contributing
# Everything lands in ${CMAKE_CURRENT_BINARY_DIR}/out; `cmake --build <dir> --target speech_docs`
# builds it all, `speech_docs_cpp` / `speech_docs_python` build one part.

cmake_minimum_required(VERSION 3.19)

set(SPEECH_DOCS_OUT ${CMAKE_CURRENT_BINARY_DIR}/out)
set(SPEECH_DOCS_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
set(SPEECH_DOCS_CPP_OUT ${SPEECH_DOCS_OUT}/cpp)
set(SPEECH_DOCS_PYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)
set(SPEECH_DOCS_PYTHON_OUT ${SPEECH_DOCS_OUT}/python)
set(SPEECH_DOCS_LOGO ${PROJECT_SOURCE_DIR}/docs/logo/libspeech-logo.svg)

# ── C++ documentation (Doxygen) ───────────────────────────────
find_package(Doxygen)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(DOXYGEN_FOUND)
    if(DOXYGEN_DOT_FOUND)
        set(SPEECH_DOCS_HAVE_DOT YES)
    else()
        set(SPEECH_DOCS_HAVE_DOT NO)
    endif()
    set(SPEECH_DOCS_CPP_HTML ${SPEECH_DOCS_CPP_OUT}/html/index.html)
    file(GLOB_RECURSE SPEECH_DOCS_CPP_HEADERS ${PROJECT_SOURCE_DIR}/include/libspeech/*.h)

    add_custom_command(
        OUTPUT ${SPEECH_DOCS_CPP_HTML}
        COMMAND ${CMAKE_COMMAND}
            -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DPROJECT_VERSION=${PROJECT_VERSION}
            -DOUTPUT_DIR=${SPEECH_DOCS_CPP_OUT}
            -DDOXYGEN_EXECUTABLE=${DOXYGEN_EXECUTABLE}
            -DHAVE_DOT=${SPEECH_DOCS_HAVE_DOT}
            -DFOOTER_IN=${SPEECH_DOCS_CPP_DIR}/footer.html.in
            -DDOXYFILE_IN=${SPEECH_DOCS_CPP_DIR}/Doxyfile.in
            -DLOGO_SRC=${SPEECH_DOCS_LOGO}
            -P ${SPEECH_DOCS_CPP_DIR}/CreateDocs.cmake
        DEPENDS
            ${SPEECH_DOCS_CPP_DIR}/CreateDocs.cmake
            ${SPEECH_DOCS_CPP_DIR}/footer.html.in
            ${SPEECH_DOCS_CPP_DIR}/Doxyfile.in
            ${SPEECH_DOCS_CPP_DIR}/mainpage.md
            ${SPEECH_DOCS_CPP_HEADERS}
        COMMENT "Generating C++ API documentation with Doxygen"
        VERBATIM
    )
    add_custom_target(speech_docs_cpp DEPENDS ${SPEECH_DOCS_CPP_HTML})
else()
    message(WARNING "Doxygen not found - C++ documentation will not be generated")
    add_custom_target(speech_docs_cpp
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping C++ docs: Doxygen not found")
endif()

# ── Python documentation (Sphinx) ─────────────────────────────
# Sphinx imports the package staged in the build tree (see src/bindings/python/CMakeLists.txt),
# so it needs BUILD_PYTHON and the `speech_stubs` target (which depends on every module) built first.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sphinx"
    OUTPUT_QUIET ERROR_QUIET
    RESULT_VARIABLE SPHINX_CHECK)

if(SPHINX_CHECK EQUAL 0 AND DEFINED PYTHON_PROJECT_NAME AND NOT DEFINED SKBUILD)
    set(SPEECH_DOCS_PYTHON_HTML ${SPEECH_DOCS_PYTHON_OUT}/html/index.html)
    file(GLOB SPEECH_DOCS_PYTHON_SOURCES ${SPEECH_DOCS_PYTHON_DIR}/source/*.rst)

    add_custom_command(
        OUTPUT ${SPEECH_DOCS_PYTHON_HTML}
        COMMAND ${CMAKE_COMMAND}
            -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
            -DSOURCE_DIR=${SPEECH_DOCS_PYTHON_DIR}/source
            -DOUTPUT_DIR=${SPEECH_DOCS_PYTHON_OUT}
            -DSPEECH_PYTHON_INSTALL_DIR=${PYTHON_PROJECT_NAME}
            -P ${SPEECH_DOCS_PYTHON_DIR}/CreateDocs.cmake
        DEPENDS
            ${SPEECH_DOCS_PYTHON_DIR}/CreateDocs.cmake
            ${SPEECH_DOCS_PYTHON_DIR}/source/conf.py
            ${SPEECH_DOCS_PYTHON_SOURCES}
        COMMENT "Building Python API documentation with Sphinx"
        VERBATIM
    )
    add_custom_target(speech_docs_python DEPENDS ${SPEECH_DOCS_PYTHON_HTML})
    # defined later in the top-level CMakeLists.txt (src/bindings/python): resolved at generate time
    add_dependencies(speech_docs_python speech_stubs)
else()
    message(WARNING "Python docs skipped: need Sphinx (pip install sphinx), BUILD_PYTHON and a "
                    "local (non-wheel) build")
    add_custom_target(speech_docs_python
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping Python docs")
endif()

# ── Route index page ──────────────────────────────────────────
set(SPEECH_INDEX_OUT ${SPEECH_DOCS_OUT}/index.html)
set(SPEECH_SUPPORTED_LANGS ${CMAKE_CURRENT_SOURCE_DIR}/supported_langs.json)
set(SPEECH_GENERATE_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/generate_index.py)
# Globbed at configure time: a new fragment file needs one re-run of cmake to be picked up.
file(GLOB_RECURSE SPEECH_LANG_FRAGMENTS
    ${CMAKE_CURRENT_SOURCE_DIR}/*/index-docs.json
    ${CMAKE_CURRENT_SOURCE_DIR}/*/index-install.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/*/index-example.txt)

add_custom_command(
    OUTPUT ${SPEECH_INDEX_OUT}
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT_DIR=${SPEECH_DOCS_OUT}
        -DINDEX_IN=${CMAKE_CURRENT_SOURCE_DIR}/index.html.in
        -DCSS_FILE=${CMAKE_CURRENT_SOURCE_DIR}/speech-docs.css
        -DPROJECT_VERSION=${PROJECT_VERSION}
        -DLOGO_SQ=${SPEECH_DOCS_LOGO}
        -DLANGS_FILE=${SPEECH_SUPPORTED_LANGS}
        -DLANGS_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DGENERATE_SCRIPT=${SPEECH_GENERATE_SCRIPT}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/CreateDocs.cmake
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/CreateDocs.cmake
        ${SPEECH_GENERATE_SCRIPT}
        ${CMAKE_CURRENT_SOURCE_DIR}/index.html.in
        ${CMAKE_CURRENT_SOURCE_DIR}/speech-docs.css
        ${SPEECH_DOCS_LOGO}
        ${SPEECH_SUPPORTED_LANGS}
        ${SPEECH_LANG_FRAGMENTS}
    COMMENT "Creating documentation route index"
    VERBATIM
)
add_custom_target(speech_docs_index DEPENDS ${SPEECH_INDEX_OUT})

# ── Markdown pages (SpeechPages) ──────────────────────────────
set(SPEECH_DOCS_SRC ${CMAKE_CURRENT_SOURCE_DIR})
set(LOGO_SQ ${SPEECH_DOCS_LOGO})
include(${CMAKE_CURRENT_SOURCE_DIR}/SpeechPages.cmake)

speech_add_page(NAME changelog  SOURCE ${PROJECT_SOURCE_DIR}/CHANGELOG.md     TITLE "Changelog")
speech_add_page(NAME renames    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/api-renames.md TITLE "C++ API renames" NAV_LABEL "Renames")
speech_add_page(NAME contributing SOURCE ${PROJECT_SOURCE_DIR}/CONTRIBUTING.md TITLE "Contributing")

# ── Master target ─────────────────────────────────────────────
add_custom_target(speech_docs
    COMMAND ${CMAKE_COMMAND} -E echo "-- All documentation generated"
    COMMENT "Building all documentation")
add_dependencies(speech_docs speech_docs_cpp speech_docs_python speech_docs_index)
speech_finalize_pages(speech_docs)

# Benchmark report: generated on a developer machine and committed, only copied here.
set(SPEECH_BENCH_REPORT ${PROJECT_SOURCE_DIR}/benchmarks/results/report.html)
if(EXISTS ${SPEECH_BENCH_REPORT})
    add_custom_command(TARGET speech_docs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SPEECH_DOCS_OUT}/benchmarks
        COMMAND ${CMAKE_COMMAND} -E copy ${SPEECH_BENCH_REPORT} ${SPEECH_DOCS_OUT}/benchmarks/index.html
        COMMENT "Publishing the committed benchmark report"
        VERBATIM)
endif()

# Keep or drop the footer links to the benchmark/coverage pages, copy a coverage dashboard from
# a coverage build tree if there is one next to this build (see FinishPages.cmake)
add_custom_command(TARGET speech_docs POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT_DIR=${SPEECH_DOCS_OUT}
        -DCOVERAGE_DIR=${CMAKE_BINARY_DIR}/coverage
        -P ${CMAKE_CURRENT_SOURCE_DIR}/FinishPages.cmake
    COMMENT "Finishing documentation pages"
    VERBATIM)

# Inject the "generated by" comment into every HTML page once everything is built
add_custom_command(TARGET speech_docs POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT_DIR=${SPEECH_DOCS_OUT}
        -DCOMMENT_IN=${CMAKE_CURRENT_SOURCE_DIR}/initial_comment.in
        -DASCII_ART_FILE=${CMAKE_CURRENT_SOURCE_DIR}/images/libspeech-logo.txt
        -DPROJECT_VERSION=${PROJECT_VERSION}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/InjectComment.cmake
    COMMENT "Injecting HTML comments into documentation pages"
    VERBATIM)
