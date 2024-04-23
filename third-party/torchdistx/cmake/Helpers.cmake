# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

function(torchdist_add_target target)
    cmake_parse_arguments(arg
        #OPTIONS
            "EXECUTABLE;LIBRARY;SHARED_LIBRARY;STATIC_LIBRARY;PYTHON_MODULE"
        #KEYWORDS
            "OUTPUT_NAME"
        #MULTI_VALUE_KEYWORDS
            ""
        #ARGUMENTS
            ${ARGN}
    )

    if(arg_EXECUTABLE)
        add_executable(${target})
    elseif(arg_PYTHON_MODULE)
        if(NOT COMMAND Python3_add_library)
            message(FATAL_ERROR "Python3 must be loaded before calling torchdist_add_target()!")
        endif()

        Python3_add_library(${target} WITH_SOABI)
    else()
        if(arg_LIBRARY)
            set(lib_type)
        elseif(arg_SHARED_LIBRARY)
            set(lib_type SHARED)
        elseif(arg_STATIC_LIBRARY)
            set(lib_type STATIC)
        else()
            message(FATAL_ERROR "torchdist_add_target() has an invalid target type!")
        endif()

        add_library(${target} ${lib_type})
    endif()

    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR
        PARENT_PATH
            source_parent_dir
    )

    if(arg_LIBRARY OR arg_SHARED_LIBRARY OR arg_STATIC_LIBRARY)
        if(PROJECT_IS_TOP_LEVEL)
            set(system)
        else()
            set(system SYSTEM)
        endif()

        target_include_directories(${target} ${system}
            INTERFACE
                $<BUILD_INTERFACE:${source_parent_dir}>
        )
    endif()

    # ------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------

    set_target_properties(${target} PROPERTIES
        C_EXTENSIONS
            OFF
        C_VISIBILITY_PRESET
            hidden
        CXX_EXTENSIONS
            OFF
        CXX_VISIBILITY_PRESET
            hidden
        CUDA_EXTENSIONS
            OFF
        CUDA_VISIBILITY_PRESET
            hidden
        POSITION_INDEPENDENT_CODE
            ON
        EXPORT_COMPILE_COMMANDS
            ON
    )

    if(arg_SHARED_LIBRARY AND NOT TORCHDIST_INSTALL_STANDALONE)
        set_target_properties(${target} PROPERTIES
            VERSION
                ${PROJECT_VERSION}
            SOVERSION
                ${PROJECT_VERSION_MAJOR}
        )
    endif()

    if(arg_OUTPUT_NAME)
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME
                ${arg_OUTPUT_NAME}
        )
    endif()

    if(TORCHDIST_PERFORM_LTO)
        set_target_properties(${target} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION
                ON
        )

        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            torchdist_set_macos_lto_path(${target})
        endif()
    endif()

    if(arg_PYTHON_MODULE AND TORCHDIST_DEVELOP_PYTHON)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH_USE_ORIGIN
                OFF
        )

        add_custom_command(
            TARGET
                ${target}
            POST_BUILD
            COMMAND
                ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${target}>" "${source_parent_dir}"
            VERBATIM
        )
    endif()

    torchdist_enable_clang_tidy(${target})

    # ------------------------------------------------------------
    # Compiler Settings
    # ------------------------------------------------------------

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target}
            PRIVATE
                -fasynchronous-unwind-tables -fstack-protector-strong
        )

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
                message(FATAL_ERROR "Only GCC 7 and later versions are supported!")
            endif()

            target_compile_options(${target}
                PRIVATE
                    -Wall
                    -Wcast-align
                    -Wconversion
                    -Wdouble-promotion
                    -Wextra
                    -Wfloat-equal
                    -Wformat=2
                    -Winit-self
                    -Wlogical-op
                    -Wno-unknown-pragmas
                    -Wpointer-arith
                    -Wshadow
                    -Wsign-conversion
                    -Wswitch-enum
                    -Wunused
                    $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wold-style-cast>
                    $<$<COMPILE_LANGUAGE:CXX>:-Woverloaded-virtual>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wuseless-cast>
            )

            target_compile_definitions(${target}
                PRIVATE
                    $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
            )
        else()
            if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
                message(FATAL_ERROR "Only Clang 7 and later versions are supported!")
            endif()

            target_compile_options(${target}
                PRIVATE
                    -fsized-deallocation
                    -Weverything
                    -Wno-c++98-compat
                    -Wno-c++98-compat-pedantic
                    -Wno-exit-time-destructors
                    -Wno-extra-semi-stmt
                    -Wno-global-constructors
                    -Wno-padded
                    -Wno-return-std-move-in-c++11
                    -Wno-shadow-uncaptured-local
            )
        endif()

        if(TORCHDIST_TREAT_WARNINGS_AS_ERRORS)
            target_compile_options(${target}
                PRIVATE
                    -Werror
            )
        endif()

        if(TORCHDIST_BUILD_FOR_NATIVE)
            target_compile_options(${target}
                PRIVATE
                    -march=native -mtune=native
            )
        endif()

        target_compile_definitions(${target}
            PRIVATE
                $<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>
        )
    else()
        message(FATAL_ERROR "Only GCC and Clang toolchains are supported!")
    endif()

    # ------------------------------------------------------------
    # Linker Settings
    # ------------------------------------------------------------

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(${target}
            PRIVATE
                LINKER:--as-needed
                LINKER:--build-id=sha1
                LINKER:-z,noexecstack
                LINKER:-z,now
                LINKER:-z,relro
        )

        if(NOT arg_PYTHON_MODULE)
            target_link_options(${target}
                PRIVATE
                    LINKER:-z,defs
            )
        endif()

        if(TORCHDIST_TREAT_WARNINGS_AS_ERRORS)
            target_link_options(${target}
                PRIVATE
                    LINKER:--fatal-warnings
            )
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_link_options(${target}
            PRIVATE
                LINKER:-bind_at_load
        )

        if(arg_PYTHON_MODULE)
            target_link_options(${target}
                PRIVATE
                    LINKER:-undefined,dynamic_lookup
            )
        else()
            target_link_options(${target}
                PRIVATE
                    LINKER:-undefined,error
            )
        endif()

        # Conda Build sets the `-pie` option in `LDFLAGS` which causes a linker warning for library
        # targets. When warnings are treated as errors, this becomes a build failure.
        if(NOT arg_EXECUTABLE)
            target_link_options(${target}
                PRIVATE
                    LINKER:-no_pie
            )
        endif()

        if(TORCHDIST_TREAT_WARNINGS_AS_ERRORS)
            target_link_options(${target}
                PRIVATE
                    LINKER:-fatal_warnings
            )
        endif()
    else()
        message(FATAL_ERROR "Only Linux and macOS operating systems are supported!")
    endif()

    # ------------------------------------------------------------
    # Sanitizers
    # ------------------------------------------------------------

    if(TORCHDIST_SANITIZERS)
        string(TOLOWER "${TORCHDIST_SANITIZERS}"
            #OUTPUT
                sanitizer_types
        )

        foreach(sanitizer_type IN ITEMS ${sanitizer_types})
            if(sanitizer_type STREQUAL "asan")
                if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                    target_compile_definitions(${target}
                        PRIVATE
                            _GLIBCXX_SANITIZE_VECTOR
                    )
                endif()

                list(APPEND sanitizers -fsanitize=address)
            elseif(sanitizer_type STREQUAL "ubsan")
                list(APPEND sanitizers -fsanitize=undefined)
            elseif(sanitizer_type STREQUAL "tsan")
                list(APPEND sanitizers -fsanitize=thread)
            else()
                message(FATAL_ERROR "The specified sanitizer type is invalid!")
            endif()
        endforeach()

        target_compile_options(${target}
            PRIVATE
                ${sanitizers} -fno-omit-frame-pointer
        )

        target_link_options(${target}
            PRIVATE
                ${sanitizers}
        )
    endif()
endfunction()

# When performing ThinLTO on macOS, mach-o object files are generated under a
# temporary directory that gets deleted by the linker at the end of the build
# process. Thus tools such as dsymutil cannot access the DWARF info contained
# in those files. To ensure that the object files still exist after the build
# process we have to set the `object_path_lto` linker option.
function(torchdist_set_macos_lto_path target)
    get_target_property(
        #OUT
            target_type
        #TARGET
            ${target}
        #PROPERTY
            TYPE
    )

    if(target_type STREQUAL "STATIC_LIBRARY")
        return()
    endif()

    set(lto_dir ${CMAKE_CURRENT_BINARY_DIR}/lto.d/${target}/${CMAKE_CFG_INTDIR})

    add_custom_command(
        TARGET
            ${target}
        PRE_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${lto_dir}"
        VERBATIM
    )

    # See man ld(1).
    target_link_options(${target}
        PRIVATE
            LINKER:-object_path_lto "${lto_dir}"
    )

    set_property(DIRECTORY APPEND PROPERTY
        ADDITIONAL_MAKE_CLEAN_FILES
            ${lto_dir}
    )
endfunction()

function(torchdist_add_third_party)
    foreach(project IN ITEMS ${ARGV})
        add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third-party/${project} EXCLUDE_FROM_ALL)
    endforeach()
endfunction()

function(torchdist_enable_clang_tidy)
    if(NOT TORCHDIST_RUN_CLANG_TIDY)
        return()
    endif()

    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "clang-tidy can only be used with the Clang toolchain!")
    endif()

    find_program(TORCHDIST_CLANG_TIDY_PROG NAMES clang-tidy REQUIRED)

    mark_as_advanced(TORCHDIST_CLANG_TIDY_PROG)

    foreach(target IN ITEMS ${ARGV})
        set_target_properties(${target} PROPERTIES
            C_CLANG_TIDY
                ${TORCHDIST_CLANG_TIDY_PROG}
            CXX_CLANG_TIDY
                ${TORCHDIST_CLANG_TIDY_PROG}
            CUDA_CLANG_TIDY
                ${TORCHDIST_CLANG_TIDY_PROG}
        )
    endforeach()
endfunction()

function(torchdist_install target)
    cmake_parse_arguments(arg "" "PACKAGE" "HEADERS" ${ARGN})

    # Set rpath if we are installing in standalone mode.
    if(TORCHDIST_INSTALL_STANDALONE)
        set(install_bindir bin)
        set(install_libdir lib)

        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            set(rpath_origin @loader_path)
        else()
            set(rpath_origin \$ORIGIN)
        endif()

        get_target_property(
            #OUT
                target_type
            #TARGET
                ${target}
            #PROPERTY
                TYPE
        )

        if(target_type STREQUAL "EXECUTABLE")
            set(target_rpath ${rpath_origin}/../lib)
        else()
            set(target_rpath ${rpath_origin})
        endif()

        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH
                ${target_rpath}
        )
    else()
        set(install_bindir ${CMAKE_INSTALL_BINDIR})
        set(install_libdir ${CMAKE_INSTALL_LIBDIR})
    endif()

    install(
        TARGETS
            ${target}
        EXPORT
            ${arg_PACKAGE}-targets
        RUNTIME
            DESTINATION
                ${install_bindir}
            COMPONENT
                runtime
        LIBRARY
            DESTINATION
                ${install_libdir}
            COMPONENT
                runtime
            NAMELINK_COMPONENT
                devel
        ARCHIVE
            DESTINATION
                ${install_libdir}
            COMPONENT
                devel
        INCLUDES DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR}
    )

    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR
        PARENT_PATH
            source_parent_dir
    )

    foreach(header IN ITEMS ${arg_HEADERS})
        cmake_path(REMOVE_FILENAME header
            OUTPUT_VARIABLE
                relative_header_dir
        )

        set(header_dir ${CMAKE_CURRENT_SOURCE_DIR}/${relative_header_dir})

        cmake_path(RELATIVE_PATH header_dir
            BASE_DIRECTORY
                ${source_parent_dir}
        )

        install(
            FILES
                ${header}
            DESTINATION
                ${CMAKE_INSTALL_INCLUDEDIR}/${header_dir}
            COMPONENT
                devel
        )
    endforeach()
endfunction()

function(torchdist_install_python_module target)
    # Set rpath if we are installing in standalone mode.
    if(TORCHDIST_INSTALL_STANDALONE)
        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            set(rpath_origin @loader_path)
        else()
            set(rpath_origin \$ORIGIN)
        endif()

        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH
                ${rpath_origin}/lib
        )
    endif()

    install(
        TARGETS
            ${target}
        LIBRARY
            DESTINATION
                .
            COMPONENT
                python
        EXCLUDE_FROM_ALL
    )
endfunction()

function(torchdist_install_package package config_file)
    if(TORCHDIST_INSTALL_STANDALONE)
        set(install_libdir lib)
    else()
        set(install_libdir ${CMAKE_INSTALL_LIBDIR})
    endif()

    set(package_dir ${install_libdir}/cmake/${package}-${PROJECT_VERSION})

    configure_package_config_file(
        #INPUT
            ${config_file}
        #OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/${package}/lib/cmake/${package}/${package}-config.cmake
        INSTALL_DESTINATION
            ${package_dir}
        NO_SET_AND_CHECK_MACRO
    )

    write_basic_package_version_file(
        #OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/${package}/lib/cmake/${package}/${package}-config-version.cmake
        VERSION
            ${PROJECT_VERSION}
        COMPATIBILITY
            AnyNewerVersion
    )

    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/${package}/lib/cmake/${package}/${package}-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/${package}/lib/cmake/${package}/${package}-config-version.cmake
        DESTINATION
            ${package_dir}
        COMPONENT
            devel
    )

    install(
        EXPORT
            ${package}-targets
        FILE
            ${package}-targets.cmake
        DESTINATION
            ${package_dir}
        COMPONENT
            devel
        NAMESPACE
            ${package}::
    )

    export(
        EXPORT
            ${package}-targets
        FILE
            ${CMAKE_CURRENT_BINARY_DIR}/${package}/lib/cmake/${package}/${package}-targets.cmake
        NAMESPACE
            ${package}::
    )
endfunction()
