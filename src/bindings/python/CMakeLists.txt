# Python bindings (nanobind), laid out like ctoon's src/bindings/python/CMakeLists.txt:
# every nanobind module is a bind_*.cpp next to this file, the pure-Python package lives in
# ./libspeech/, and the .pyi type stubs are generated here and installed with the package.
# Included from the top-level CMakeLists.txt when PYTHON_PROJECT_NAME is defined.

find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

set(NB_MODULE _about)
nanobind_add_module(${NB_MODULE} STABLE_ABI NB_STATIC ${PROJECT_SOURCE_DIR}/src/bindings/python/about.cpp)
libspeech_set_origin_rpath(${NB_MODULE})

# __init__.py's cdll.LoadLibrary() calls need the *exact* on-disk
# filename of each preloaded runtime library (speech itself, httpp,
# ONNXRuntime) -- these vary by platform (libspeech.so / speech.dll /
# libspeech.dylib, etc.) and, for ONNXRuntime, by version too. RPATH
# (see libspeech_set_origin_rpath above) can't help with any of this:
# it only lets the OS loader resolve a DT_NEEDED entry when one
# already-loaded .so pulls in another at normal link time. ctypes'
# cdll.LoadLibrary() is a direct, explicit dlopen()/LoadLibrary() call
# from Python itself, entirely bypassing that mechanism -- it needs the
# literal filename handed to it, RPATH or not.
#
# Rather than have __init__.py guess that filename per-platform (with
# if/else branches or fragile globs), compute it once, here, where
# CMake already knows it precisely, and bake it into _about (already a
# tiny, dependency-free nanobind module that __init__.py imports first,
# for __version__) as extra module attributes -- alongside __version__,
# the same way this file already bakes VERSION_INFO in below.
# $<TARGET_FILE_NAME:...> is a generator expression, resolved once CMake
# knows the real output name of each target.
# On Linux, ship (and load) the library under its SONAME
# (libonnxruntime.so.1) rather than the fully-versioned filename
# (libonnxruntime.so.1.21.0): both names used to be installed as full
# ~20MB copies (the SONAME file is a symlink on the upstream release
# tarball, but CMake's install(FILES ...)/file(COPY ...) dereference
# symlinks, and it turns out the wheel-building step also flattens an
# install-time symlink back into a second real copy when zipping -- so
# there is no way to keep both names *and* avoid shipping the bytes
# twice). Since the SONAME is the one name every consumer .so's
# DT_NEEDED entry actually requires at runtime, standardize the
# explicit ctypes preload in __init__.py on that same name too and
# stop installing the fully-versioned name altogether -- one file,
# one name, used both places. Windows/macOS have no such SONAME split
# (see cmake/ONNXRuntime.cmake) so they keep using ONNXRUNTIME_LIB_FILE.
if(ONNXRUNTIME_SONAME_FILE)
    get_filename_component(_ONNXRUNTIME_LIB_FILENAME "${ONNXRUNTIME_SONAME_FILE}" NAME)
else()
    get_filename_component(_ONNXRUNTIME_LIB_FILENAME "${ONNXRUNTIME_LIB_FILE}" NAME)
endif()
target_compile_definitions(${NB_MODULE} PRIVATE
        VERSION_INFO=${PROJECT_VERSION}
        NB_MODULE_NAME=${NB_MODULE}
        LIB_PATH=$<TARGET_FILE_NAME:speech>
        HTTPP_LIB_PATH=$<TARGET_FILE_NAME:httpp::httpp_core>
        ONNXRUNTIME_LIB_PATH=${_ONNXRUNTIME_LIB_FILENAME})

# CONFIGURE_DEPENDS: best-effort auto-detection of new bind_*.cpp files
# without a manual reconfigure. This is NOT guaranteed by CMake to work
# on every generator/environment (e.g. observed unreliable with the Unix
# Makefiles generator in some setups) -- if a newly added binding module
# doesn't show up after `cmake --build`, run `cmake ..` (or delete and
# recreate the build directory) once to force a fresh configure.
file(GLOB PYTHON_BIND_MODULES_PATH CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/bindings/python/bind_*.cpp)
set(PYTHON_BIND_MODULES "${NB_MODULE}")
foreach(NB_MODULE_SOURCE IN LISTS PYTHON_BIND_MODULES_PATH)
    get_filename_component(NB_MODULE ${NB_MODULE_SOURCE} NAME_WE)
    string(REGEX REPLACE "bind_(.*)" "speech_\\1_py" NB_MODULE ${NB_MODULE})
    list(APPEND PYTHON_BIND_MODULES ${NB_MODULE})

    message(STATUS "Found module: ${NB_MODULE}")

    # We are now ready to compile the actual extension module
    nanobind_add_module(
            # Name of the extension
            ${NB_MODULE}

            # Target the stable ABI for Python 3.12+, which reduces
            # the number of binary wheels that must be built. This
            # does nothing on older Python versions
            STABLE_ABI

            # Build libnanobind statically and merge it into the
            # extension (which itself remains a shared library)
            #
            # If your project builds multiple extensions, you can
            # replace this flag by NB_SHARED to conserve space by
            # reusing a shared libnanobind across libraries
            NB_STATIC

            # Source code goes here -- one file per module (ctoon's
            # convention: src/bindings/python/bind_X.cpp -> speech_X_py),
            # unlike _about above which is a single nanobind_add_module
            # call named directly rather than discovered by this glob.
            ${NB_MODULE_SOURCE}
    )

    target_include_directories(${NB_MODULE} PRIVATE include src)
    target_compile_definitions(${NB_MODULE} PRIVATE NB_MODULE_NAME=${NB_MODULE})
    target_link_libraries(${NB_MODULE} PRIVATE speech::speech)
    libspeech_set_origin_rpath(${NB_MODULE})
endforeach()

# Stage the package (python files, ONNXRuntime, compiled modules and speech) in one directory of
# the build tree, so the modules can be imported there to generate their stubs. A local build
# stages straight into PYTHON_PROJECT_NAME (the tests run from there); a wheel build uses a
# private directory and installs from the real build outputs.
if(DEFINED SKBUILD)
    set(_stage_root ${CMAKE_BINARY_DIR}/stage)
else()
    set(_stage_root ${PYTHON_PROJECT_NAME})
endif()
if(DEFINED SKBUILD)
    # `speech` (line ~221: add_library(speech SHARED ...)) is a real
    # SHARED library, not a MODULE (unlike ${PYTHON_BIND_MODULES}, the
    # nanobind extension .pyd/.so files, which install fine under
    # LIBRARY alone on every platform). CMake's install(TARGETS) docs
    # spell out the platform split this causes: "For DLL platforms
    # (Windows) the DLL part of a shared library is treated as a
    # RUNTIME target, while the corresponding import library is
    # treated as an ARCHIVE target. On non-DLL platforms shared
    # libraries are treated as LIBRARY targets." LIBRARY DESTINATION
    # alone is therefore complete on Linux/macOS but, on Windows,
    # speech.dll itself belongs to a target kind (RUNTIME) this rule
    # never mentions -- so it silently isn't installed at all: no
    # error, no warning, just a wheel with every OTHER file in place
    # and speech.dll simply missing, which is indistinguishable at
    # runtime from "installed but failed to load one of its
    # dependencies" (identical ctypes error either way) until you
    # actually go looking for the file.
    install (TARGETS ${PYTHON_BIND_MODULES} speech
            CONFIGURATIONS Release
            LIBRARY DESTINATION ${PYTHON_PROJECT_NAME}
            RUNTIME DESTINATION ${PYTHON_PROJECT_NAME})

    # The pure-Python files (currently just __init__.py) need
    # their own explicit install rule -- scikit-build-core's automatic
    # Python-package discovery only looks for an immediate src/<name>
    # layout, so now that the package lives nested under
    # src/bindings/python/${PROJECT_NAME} (ctoon's own convention,
    # matched here on request), it no longer finds it on its own: the
    # compiled extensions above still land in the wheel fine (that
    # goes through install(TARGETS), unrelated to this discovery), but
    # without this, the wheel would ship no __init__.py at all --
    # `import libspeech` would "succeed" only as an empty implicit
    # namespace package, silently missing everything real. Same
    # pattern ctoon itself uses for exactly this reason (see its
    # src/bindings/python/CMakeLists.txt).
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/bindings/python/${PROJECT_NAME}/
            DESTINATION ${SKBUILD_PROJECT_NAME}
            FILES_MATCHING PATTERN "*.py" PATTERN "py.typed")

    # ONNXRuntime ships as a shared library that the compiled extension
    # modules dlopen symbols from at import time (see __init__.py's
    # cdll.LoadLibrary() calls), so it has to be copied into the wheel
    # next to those extensions rather than relied upon via RPATH:
    # RPATH would only point back at wherever it got downloaded to on
    # the *build* machine, which has no reason to exist on whatever
    # machine later installs this wheel.
    #
    # httpp is NOT bundled here -- __init__.py loads it from the
    # separately pip-installed httpp package instead (see __init__.py).
    #
    # ONNXRUNTIME_SONAME_FILE (see cmake/ONNXRuntime.cmake) is the name
    # every compiled .so's DT_NEEDED entry actually requires at
    # runtime (libonnxruntime.so.1 on Linux), not the fully-versioned
    # ONNXRUNTIME_LIB_FILE one -- without a file by that exact name,
    # `auditwheel repair` fails to find/bundle it at all.
    #
    # On disk (in the upstream release tarball) that name is a
    # symlink to ONNXRUNTIME_LIB_FILE, which is why it used to look
    # free to ship both: but CMake's install(FILES ...) dereferences
    # symlinks, AND the wheel-building step separately flattens an
    # install-time symlink back into a real copy when zipping -- so
    # shipping *both* names always costs a full second ~20MB copy no
    # matter which way the symlink is recreated. Since the SONAME is
    # the only name anything actually needs to dlopen/link against at
    # runtime (see the ONNXRUNTIME_LIB_PATH comment above), install
    # the real bytes under that name ONLY and drop the fully-versioned
    # name from the wheel entirely -- one ~20MB file instead of two.
    # ONNXRUNTIME_PROVIDERS_SHARED_FILE (Windows only) is the companion
    # onnxruntime_providers_shared.dll upstream ships alongside
    # onnxruntime.dll -- see cmake/ONNXRuntime.cmake for why.
    if(ONNXRUNTIME_SONAME_FILE)
        get_filename_component(_ONNXRUNTIME_SONAME_FILENAME "${ONNXRUNTIME_SONAME_FILE}" NAME)
        # RENAME so only one copy of the (large) file is ever written
        # -- under the SONAME, not the fully-versioned name.
        install(FILES "${ONNXRUNTIME_LIB_FILE}"
                DESTINATION ${SKBUILD_PROJECT_NAME}
                RENAME "${_ONNXRUNTIME_SONAME_FILENAME}")
    else()
        install(FILES "${ONNXRUNTIME_LIB_FILE}"
                DESTINATION ${SKBUILD_PROJECT_NAME})
    endif()
    if(ONNXRUNTIME_PROVIDERS_SHARED_FILE)
        install(FILES "${ONNXRUNTIME_PROVIDERS_SHARED_FILE}"
                DESTINATION ${SKBUILD_PROJECT_NAME})
    endif()

    # Type stubs generated below (speech_stubs) ship with the package, like ctoon's ctoon_py.pyi.
    foreach(_module IN LISTS PYTHON_BIND_MODULES)
        install(FILES ${_stage_root}/${PROJECT_NAME}/${_module}.pyi DESTINATION ${SKBUILD_PROJECT_NAME})
    endforeach()
endif()

if(TRUE)
    file(COPY ${PROJECT_SOURCE_DIR}/src/bindings/python/${PROJECT_NAME} DESTINATION ${_stage_root})
    if(ONNXRUNTIME_SONAME_FILE)
        # Same fix as the scikit-build install(FILES ...) branch above:
        # ship the real bytes under the SONAME only (one ~20MB file,
        # not the fully-versioned name too), since that's the only
        # name anything actually dlopens/links against at runtime.
        get_filename_component(_ONNXRUNTIME_SONAME_FILENAME "${ONNXRUNTIME_SONAME_FILE}" NAME)
        file(COPY_FILE "${ONNXRUNTIME_LIB_FILE}"
             "${_stage_root}/${PROJECT_NAME}/${_ONNXRUNTIME_SONAME_FILENAME}")
    else()
        file(COPY ${ONNXRUNTIME_LIB_FILE} DESTINATION ${_stage_root}/${PROJECT_NAME})
    endif()
    if(ONNXRUNTIME_PROVIDERS_SHARED_FILE)
        file(COPY ${ONNXRUNTIME_PROVIDERS_SHARED_FILE} DESTINATION ${_stage_root}/${PROJECT_NAME})
    endif()

    # The generator expression is deliberate: multi-config generators (Visual Studio, Xcode,
    # Ninja Multi-Config) append a per-config subdirectory (.../libspeech/Release/) to a plain
    # output-directory path, which moves the modules out of the package directory that
    # __init__.py and the stub generator import them from. `$<1:...>` is just the path, but
    # being a generator expression it switches that suffix off.
    set(_stage_pkg_dir "$<1:${_stage_root}/${PROJECT_NAME}>")

    foreach(_core IN LISTS PYTHON_BIND_MODULES)
        set_target_properties(${_core} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_stage_pkg_dir})
    endforeach()

    # speech itself must land inside the package directory too --
    # __init__.py's `cdll.LoadLibrary(_here / LIB_PATH)` looks for it
    # right next to _audio.abi3.so/_about.abi3.so, not in the
    # top-level build directory (where it would otherwise default to).
    # Both output-directory properties are needed for the same DLL-
    # platform-split reason as the SKBUILD install(TARGETS) rule above:
    # LIBRARY_OUTPUT_DIRECTORY alone places the .dll correctly on
    # Linux/macOS but is silently ignored for it on Windows, where a
    # SHARED library's DLL is governed by RUNTIME_OUTPUT_DIRECTORY
    # instead (LIBRARY_OUTPUT_DIRECTORY only affects its .lib import
    # library there) -- so on Windows the .dll would otherwise default
    # to landing directly under ${_stage_root} instead of
    # ${_stage_root}/${PROJECT_NAME}, right alongside it.
    set_target_properties(speech PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${_stage_pkg_dir}
            RUNTIME_OUTPUT_DIRECTORY ${_stage_pkg_dir})

    # Python stubs (.pyi), generated like ctoon does (python -m nanobind.stubgen), but through the
    # package (-m libspeech.<module>): the modules need the shared libraries that
    # libspeech/__init__.py preloads, so they only import from the assembled package above.
    set(_stub_files)
    set(_stub_commands)
    foreach(_module IN LISTS PYTHON_BIND_MODULES)
        list(APPEND _stub_files ${_stage_root}/${PROJECT_NAME}/${_module}.pyi)
        list(APPEND _stub_commands COMMAND ${Python_EXECUTABLE} -m nanobind.stubgen
                -m ${PROJECT_NAME}.${_module}
                -o ${_stage_root}/${PROJECT_NAME}/${_module}.pyi
                -i ${_stage_root})
    endforeach()
    add_custom_target(speech_stubs ALL
        ${_stub_commands}
        DEPENDS ${PYTHON_BIND_MODULES} speech
        BYPRODUCTS ${_stub_files}
        WORKING_DIRECTORY ${_stage_root}
        COMMENT "Generating Python stubs (.pyi) for the libspeech modules..."
        VERBATIM)
endif()

# tests/python/ needs the module list
set(PYTHON_BIND_MODULES ${PYTHON_BIND_MODULES} PARENT_SCOPE)
