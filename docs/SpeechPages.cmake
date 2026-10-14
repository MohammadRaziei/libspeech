# SpeechPages.cmake
# CMake module for building Markdown-based documentation pages.
#
# Provides:
#   speech_add_page(
#       NAME     <name>           # unique identifier, used as target name and output subdir
#       SOURCE   <file.md>        # path to the Markdown source file
#       TITLE    <"Page Title">   # displayed in <title> and page header
#       [NAV_LABEL <"Label">]     # optional short label for topbar nav (defaults to TITLE)
#       [OUTPUT_DIR <dir>]        # optional override for output dir (default: ${SPEECH_DOCS_OUT}/<name>)
#   )
#
# Each page is built as:
#   <OUTPUT_DIR>/<name>/index.html
#
# Prerequisites (must be set before including this file):
#   Python3_EXECUTABLE   — Python 3 interpreter (from find_package(Python3))
#   SPEECH_DOCS_OUT       — root output directory for all docs
#   SPEECH_DOCS_SRC       — source directory of the docs folder (contains page.html.in, speech-docs.css)
#   LOGO_SQ              — path to the SVG logo file (for embedding)
#   PROJECT_VERSION      — project version string

cmake_minimum_required(VERSION 3.19)

# ── Verify prerequisites ──────────────────────────────────────────────────────
foreach(_var Python3_EXECUTABLE SPEECH_DOCS_OUT SPEECH_DOCS_SRC PROJECT_VERSION)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "SpeechPages: ${_var} must be defined before including SpeechPages.cmake")
    endif()
endforeach()

# Check that markdown-it-py is available
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import markdown_it"
    OUTPUT_QUIET ERROR_QUIET
    RESULT_VARIABLE _MDIT_CHECK
)
if(NOT _MDIT_CHECK EQUAL 0)
    message(FATAL_ERROR
        "SpeechPages: 'markdown-it-py' Python package not found.\n"
        "Install it with:  pip install markdown-it-py"
    )
endif()

# ── Internal helper: path to the convert script ──────────────────────────────
set(_SPEECH_CONVERT_SCRIPT "${SPEECH_DOCS_SRC}/convert_page.py"
    CACHE INTERNAL "Path to the libspeech md→html conversion script")

set(_SPEECH_PAGE_TEMPLATE "${SPEECH_DOCS_SRC}/page.html.in"
    CACHE INTERNAL "Path to the libspeech page HTML template")

set(_SPEECH_PAGE_CSS "${SPEECH_DOCS_SRC}/speech-docs.css"
    CACHE INTERNAL "Path to speech-docs.css")

set(_SPEECH_SECTION_TEMPLATE "${SPEECH_DOCS_SRC}/section.html.in"
    CACHE INTERNAL "Path to the libspeech section HTML template")

# ── speech_add_page ────────────────────────────────────────────────────────────
function(speech_add_page)
    cmake_parse_arguments(ARG "" "NAME;SOURCE;TITLE;NAV_LABEL;OUTPUT_DIR" "" ${ARGN})

    if(NOT ARG_NAME OR NOT ARG_SOURCE OR NOT ARG_TITLE)
        message(FATAL_ERROR "speech_add_page: NAME, SOURCE, and TITLE are required")
    endif()

    if(NOT ARG_NAV_LABEL)
        set(ARG_NAV_LABEL "${ARG_TITLE}")
    endif()

    if(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR "${SPEECH_DOCS_OUT}/${ARG_NAME}")
    endif()

    set(_OUTPUT_HTML "${ARG_OUTPUT_DIR}/index.html")

    # Resolve absolute source path
    if(NOT IS_ABSOLUTE "${ARG_SOURCE}")
        set(ARG_SOURCE "${SPEECH_DOCS_SRC}/${ARG_SOURCE}")
    endif()

    add_custom_command(
        OUTPUT "${_OUTPUT_HTML}"
        COMMAND ${Python3_EXECUTABLE} "${_SPEECH_CONVERT_SCRIPT}"
            --input     "${ARG_SOURCE}"
            --output    "${_OUTPUT_HTML}"
            --title     "${ARG_TITLE}"
            --nav-label "${ARG_NAV_LABEL}"
            --template  "${_SPEECH_PAGE_TEMPLATE}"
            --css       "${_SPEECH_PAGE_CSS}"
            --logo      "${LOGO_SQ}"
            --version   "${PROJECT_VERSION}"
        DEPENDS
            "${ARG_SOURCE}"
            "${_SPEECH_CONVERT_SCRIPT}"
            "${_SPEECH_PAGE_TEMPLATE}"
            "${_SPEECH_PAGE_CSS}"
        COMMENT "Building page: ${ARG_TITLE}"
        VERBATIM
    )

    add_custom_target("speech_docs_page_${ARG_NAME}" DEPENDS "${_OUTPUT_HTML}")

    # Register in the global list so speech_docs can depend on all pages
    set_property(GLOBAL APPEND PROPERTY SPEECH_PAGE_TARGETS "speech_docs_page_${ARG_NAME}")

    message(STATUS "SpeechPages: registered page '${ARG_NAME}' → ${_OUTPUT_HTML}")
endfunction()

# ── speech_add_section ─────────────────────────────────────────────────────────
# Build a multi-page section where all pages share a sidebar nav + prev/next.
# Supports both .md (Markdown) and .rst (reStructuredText) source files.
#
# Usage:
#   speech_add_section(
#       NAME    <name>               # unique identifier and output subdir
#       TITLE   <"Section Title">    # shown in sidebar header and topbar
#       SOURCES                      # list of "file.md:Page Title" or "file.rst:Page Title"
#           "index.md:Overview"
#           "installation.rst:Installation"
#           "usage.md:Usage"
#           ...
#       [OUTPUT_DIR <dir>]           # optional override (default: ${SPEECH_DOCS_OUT}/<name>)
#   )
function(speech_add_section)
    cmake_parse_arguments(ARG "" "NAME;TITLE;OUTPUT_DIR" "SOURCES" ${ARGN})

    if(NOT ARG_NAME OR NOT ARG_TITLE OR NOT ARG_SOURCES)
        message(FATAL_ERROR "speech_add_section: NAME, TITLE, and SOURCES are required")
    endif()

    if(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR "${SPEECH_DOCS_OUT}/${ARG_NAME}")
    endif()

    # Build the comma-separated pages string for the Python script
    # and collect all source md files for DEPENDS
    set(_PAGES_ARG "")
    set(_MD_SOURCES "")
    foreach(_entry IN LISTS ARG_SOURCES)
        # entry is "file.md:Title" — resolve md path
        if(_entry MATCHES "^([^:]+):(.+)$")
            set(_md_file "${CMAKE_MATCH_1}")
            set(_page_title "${CMAKE_MATCH_2}")
        else()
            set(_md_file "${_entry}")
            set(_page_title "")
        endif()

        if(NOT IS_ABSOLUTE "${_md_file}")
            set(_md_file "${SPEECH_DOCS_SRC}/${_md_file}")
        endif()

        if(_PAGES_ARG)
            string(APPEND _PAGES_ARG ",${_md_file}:${_page_title}")
        else()
            set(_PAGES_ARG "${_md_file}:${_page_title}")
        endif()

        list(APPEND _MD_SOURCES "${_md_file}")
    endforeach()

    # Output stamp: we track the first page (index.html) as the primary output
    set(_STAMP "${ARG_OUTPUT_DIR}/index.html")

    add_custom_command(
        OUTPUT "${_STAMP}"
        COMMAND ${Python3_EXECUTABLE} "${_SPEECH_CONVERT_SCRIPT}"
            --section
            --section-title "${ARG_TITLE}"
            --section-dir   "${ARG_OUTPUT_DIR}"
            --section-pages "${_PAGES_ARG}"
            --source-dir    "${SPEECH_DOCS_SRC}"
            --template      "${_SPEECH_SECTION_TEMPLATE}"
            --css           "${_SPEECH_PAGE_CSS}"
            --logo          "${LOGO_SQ}"
            --version       "${PROJECT_VERSION}"
        DEPENDS
            ${_MD_SOURCES}
            "${_SPEECH_CONVERT_SCRIPT}"
            "${_SPEECH_SECTION_TEMPLATE}"
            "${_SPEECH_PAGE_CSS}"
        COMMENT "Building section: ${ARG_TITLE}"
        VERBATIM
    )

    add_custom_target("speech_docs_section_${ARG_NAME}" DEPENDS "${_STAMP}")
    set_property(GLOBAL APPEND PROPERTY SPEECH_PAGE_TARGETS "speech_docs_section_${ARG_NAME}")

    message(STATUS "SpeechPages: registered section '${ARG_NAME}' → ${ARG_OUTPUT_DIR}/")
endfunction()

# ── speech_finalize_pages ──────────────────────────────────────────────────────
macro(speech_finalize_pages TARGET)
    get_property(_page_targets GLOBAL PROPERTY SPEECH_PAGE_TARGETS)
    if(_page_targets)
        add_dependencies(${TARGET} ${_page_targets})
    endif()
endmacro()