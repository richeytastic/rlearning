cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set( CMAKE_BUILD_TYPE "Debug")
set( CMAKE_COLOR_MAKEFILE TRUE)
set( CMAKE_VERBOSE_MAKEFILE FALSE)

project(test)

set( LOCALBUILDS "$ENV{HOME}/local_builds")
set( CMAKE_MODULE_PATH "${LOCALBUILDS}/CMakeModules")
set( CMAKE_LIBRARY_PATH "${LOCALBUILDS}/libs")

set( SRC_FILES
    "${PROJECT_SOURCE_DIR}/src/test.cpp")

set( BOOST_ROOT "${LOCALBUILDS}/libs/boost")
set( Boost_USE_STATIC_LIBS ON)
set( Boost_USE_MULTITHREADED ON)
set( Boost_USE_STATIC_RUNTIME ON)
find_package( Boost 1.4 REQUIRED COMPONENTS filesystem regex system serialization thread)
include_directories( ${Boost_INCLUDE_DIRS})

set( OpenCV_DIR "${LOCALBUILDS}/libs/opencv")
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

find_package( RLearning REQUIRED)
include_directories( ${RLearning_INCLUDE_DIR})

add_executable( ${PROJECT_NAME} ${SRC_FILES})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries( ${PROJECT_NAME} ${RLearning_LIBRARY})
