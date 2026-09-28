# C++ API documentation (Doxygen), scoped to the public headers under
# include/libspeech/. Deliberately not the full multi-language docs
# machinery ctoon has (C/C++/Python/Rust/Julia/Matlab, Sphinx, a merged
# index page) -- libspeech is C++ + a 1:1 nanobind Python mirror, and the
# Python side is documented via its own docstrings, not a separate
# generated site. Revisit and extend this file if that changes.

find_package(Doxygen)

set(SPEECH_DOCS_OUT ${CMAKE_CURRENT_BINARY_DIR}/out)

if(DOXYGEN_FOUND)
    set(SPEECH_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    set(SPEECH_DOCS_HTML ${SPEECH_DOCS_OUT}/html/index.html)

    if(DOXYGEN_DOT_FOUND)
        set(_SPEECH_HAVE_DOT "YES")
    else()
        set(_SPEECH_HAVE_DOT "NO")
    endif()

    # configure_file (not ctoon's separate CreateDocs.cmake script) is
    # enough here since there's only one Doxyfile to generate, no
    # per-language footer/logo substitution.
    set(DOXYGEN_OUTPUT_DIR ${SPEECH_DOCS_OUT})
    set(DOXYGEN_HAVE_DOT ${_SPEECH_HAVE_DOT})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${SPEECH_DOXYFILE} @ONLY)

    add_custom_command(
        OUTPUT ${SPEECH_DOCS_HTML}
        COMMAND ${DOXYGEN_EXECUTABLE} ${SPEECH_DOXYFILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SPEECH_DOXYFILE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        COMMENT "Generating libspeech C++ API documentation with Doxygen"
        VERBATIM
    )

    add_custom_target(speech_docs DEPENDS ${SPEECH_DOCS_HTML})
else()
    message(WARNING "Doxygen not found - speech_docs will not be generated")
    add_custom_target(speech_docs
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping docs: Doxygen not found"
    )
endif()
