cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set( CMAKE_BUILD_TYPE "Debug")
set( CMAKE_COLOR_MAKEFILE TRUE)
set( CMAKE_VERBOSE_MAKEFILE FALSE)

project(test)

set( LOCALBUILDS "$ENV{HOME}/local_builds")
set( CMAKE_MODULE_PATH "${LOCALBUILDS}/CMakeModules")
set( CMAKE_LIBRARY_PATH "${LOCALBUILDS}/libs")

set( SRC_FILES "${PROJECT_SOURCE_DIR}/src/main.cpp")

set( FEAT_PARAMS_FILE "feat_params.cfg")
if( NOT EXISTS "${PROJECT_BINARY_DIR}/${FEAT_PARAMS_FILE}")
    file( COPY "${PROJECT_SOURCE_DIR}/${FEAT_PARAMS_FILE}" DESTINATION ${PROJECT_BINARY_DIR})
    message( STATUS "Copied across default ${FEAT_PARAMS_FILE}")
endif()

set( BOOST_ROOT "${LOCALBUILDS}/libs/boost")
set( Boost_USE_STATIC_LIBS ON)
set( Boost_USE_MULTITHREADED ON)
set( Boost_USE_STATIC_RUNTIME ON)
find_package( Boost 1.4 REQUIRED COMPONENTS filesystem regex system serialization thread)
include_directories( ${Boost_INCLUDE_DIRS})

set( OpenCV_DIR "${LOCALBUILDS}/libs/opencv")
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set( PCL_DIR "${LOCALBUILDS}/libs/PCL-1.6.0/share/pcl-1.6")
find_package( PCL 1.2 REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS})

find_package( RPascalVOC REQUIRED)
include_directories( ${RPascalVOC_INCLUDE_DIR})

find_package( RFeatures REQUIRED)
include_directories( ${RFeatures_INCLUDE_DIR})

find_package( RLearning REQUIRED)
include_directories( ${RLearning_INCLUDE_DIR})

find_package( RLIB REQUIRED)
include_directories( ${RLIB_INCLUDE_DIR})

add_executable( ${PROJECT_NAME} ${SRC_FILES})
target_link_libraries( ${PROJECT_NAME} ${RPascalVOC_LIBRARY})
target_link_libraries( ${PROJECT_NAME} ${RLearning_LIBRARY})
target_link_libraries( ${PROJECT_NAME} ${RFeatures_LIBRARY})
target_link_libraries( ${PROJECT_NAME} ${RLIB_LIBRARY})
target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries( ${PROJECT_NAME} ${PCL_LIBS})



