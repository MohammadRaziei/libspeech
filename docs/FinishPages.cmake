# FinishPages.cmake -- last step of the docs build (cmake -P).
# Required: OUTPUT_DIR. Optional: COVERAGE_DIR (the coverage dashboard of a coverage build tree,
# copied to <OUTPUT_DIR>/coverage when it exists).
#
# Footer links to pages that are generated elsewhere (benchmarks, coverage) are wrapped in
# <!--OPTIONAL:name--> ... <!--/OPTIONAL:name--> in the templates; the wrapped link is kept when
# <OUTPUT_DIR>/<name>/index.html exists and removed otherwise, so the site never links to nothing.

cmake_minimum_required(VERSION 3.19)

if(DEFINED COVERAGE_DIR AND EXISTS "${COVERAGE_DIR}/index.html")
    file(REMOVE_RECURSE "${OUTPUT_DIR}/coverage")
    file(COPY "${COVERAGE_DIR}/" DESTINATION "${OUTPUT_DIR}/coverage")
endif()

file(GLOB_RECURSE _html_files "${OUTPUT_DIR}/*.html")
foreach(_html IN LISTS _html_files)
    # generated reports (coverage, doxygen) are not ours to edit
    if(_html MATCHES "/coverage/" OR _html MATCHES "/cpp/html/" OR _html MATCHES "/benchmarks/")
        continue()
    endif()
    file(READ "${_html}" _content)
    set(_changed FALSE)
    foreach(_name benchmarks coverage)
        string(FIND "${_content}" "<!--OPTIONAL:${_name}-->" _at)
        if(_at EQUAL -1)
            continue()
        endif()
        set(_open "<!--OPTIONAL:${_name}-->")
        set(_close "<!--/OPTIONAL:${_name}-->")
        if(EXISTS "${OUTPUT_DIR}/${_name}/index.html")
            string(REPLACE "${_open}\n" "" _content "${_content}")
            string(REPLACE "${_close}\n" "" _content "${_content}")
        else()
            string(REGEX REPLACE "${_open}.*${_close}\n" "" _content "${_content}")
        endif()
        set(_changed TRUE)
    endforeach()
    if(_changed)
        file(WRITE "${_html}" "${_content}")
    endif()
endforeach()
