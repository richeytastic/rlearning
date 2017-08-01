cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set( CMAKE_BUILD_TYPE "Release")
set( CMAKE_COLOR_MAKEFILE TRUE)
set( CMAKE_VERBOSE_MAKEFILE FALSE)

project(dstats)

set( SRC_FILES
    "${PROJECT_SOURCE_DIR}/main.cpp"
    "${PROJECT_SOURCE_DIR}/../../src/PrecisionRecallFinder.cpp")

include_directories( "${PROJECT_SOURCE_DIR}/../../include")

add_executable( ${PROJECT_NAME} ${SRC_FILES})
