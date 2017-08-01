# This module defines:
#  rLearning_ROOT_DIR - the root directory where the library is installed.
#  rLearning_INCLUDE_DIR - the include directory.
#  rLearning_LIBRARY_DIR - the library directory.
#  rLearning_LIBRARY - library to link to.

get_filename_component( _prefix "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component( rLearning_ROOT_DIR "${_prefix}" PATH)

set( rLearning_INCLUDE_DIR "${rLearning_ROOT_DIR}/include")
set( rLearning_LIBRARY_DIR "${rLearning_ROOT_DIR}/lib")

include( "${CMAKE_CURRENT_LIST_DIR}/Macros.cmake")
get_library_suffix( _lib_suffix)

set( rLearning_LIBRARY rLearning${_lib_suffix})
