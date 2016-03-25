cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
project( rLearning)

include( "../LibCMakeCommon.cmake")
set( BUILD_SHARED_LIBS TRUE)

set( SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
include_directories( "${INCLUDE_DIR}")
# Include the build directory because generated header files sit here.
include_directories( "${PROJECT_BINARY_DIR}")

# Boost
if(WIN32)
    set( BOOST_ROOT "D:/boost_1_56_0")
    set( BOOST_LIBRARYDIR "${BOOST_ROOT}/lib64-msvc-11.0")
endif(WIN32)
find_package( Boost 1.5 REQUIRED COMPONENTS filesystem regex system serialization thread random)
include_directories( ${Boost_INCLUDE_DIRS})
link_directories( ${Boost_LIBRARY_DIRS})

# OpenCV
if(WIN32)
    set( OpenCV_DIR "${LOCALLIBS}/opencv")
endif(WIN32)
find_package( OpenCV 2.4 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set( RFeatures_DIR "${LOCALLIBS}/rFeatures")
find_package( RFeatures REQUIRED)
include_directories( ${RFeatures_INCLUDE_DIR})

set( LibLAS_DIR "${LOCALLIBS}/libLAS-1.7.0")
find_package( LibLAS REQUIRED)

set( RLIB_DIR "${LOCALLIBS}/rlib")
find_package( RLIB REQUIRED)
include_directories( ${RLIB_INCLUDE_DIR})

set( INCLUDE_FILES
    "${INCLUDE_DIR}/Classification.h"
    "${INCLUDE_DIR}/CrossValidator.h"
    "${INCLUDE_DIR}/CvModel.h"
    "${INCLUDE_DIR}/DataUtils.h"
    "${INCLUDE_DIR}/DecisionTreeRandomCrossValidator.h"
    "${INCLUDE_DIR}/DiscreteNaiveBayes.h"
    "${INCLUDE_DIR}/FeatureDetector.h"
    "${INCLUDE_DIR}/GaussianMAPEstimator.h"
    "${INCLUDE_DIR}/KernelCache.h"
    "${INCLUDE_DIR}/template/KernelCache_template.h"
    "${INCLUDE_DIR}/KernelFunc.h"
    "${INCLUDE_DIR}/KMeans.h"
    "${INCLUDE_DIR}/KNearestClassifier.h"
    "${INCLUDE_DIR}/KNearestNFoldCrossValidator.h"
    "${INCLUDE_DIR}/KNearestRandomCrossValidator.h"
    "${INCLUDE_DIR}/MAPEstimator.h"
    "${INCLUDE_DIR}/Model.h"
    "${INCLUDE_DIR}/template/Model_template.h"
    "${INCLUDE_DIR}/NaiveBayesRandomCrossValidator.h"
    "${INCLUDE_DIR}/NFoldCrossValidator.h"
    "${INCLUDE_DIR}/ObjectClassificationStatsManager.h"
    "${INCLUDE_DIR}/ObjectDetectionStatsManager.h"
    "${INCLUDE_DIR}/ObjectDetectionStatsManager_ViewInfo.h"
    "${INCLUDE_DIR}/PCA.h"
    "${INCLUDE_DIR}/PrecisionRecallFinder.h"
    "${INCLUDE_DIR}/RandomCrossValidator.h"
    "${INCLUDE_DIR}/RangePartsDetector.h"
    "${INCLUDE_DIR}/RealObjectSizeResponseSuppressor.h"
    "${INCLUDE_DIR}/RLearning.h"
    "${INCLUDE_DIR}/StatsGenerator.h"
    "${INCLUDE_DIR}/SVMClassifier.h"
    "${INCLUDE_DIR}/SVMNFoldCrossValidator.h"
    "${INCLUDE_DIR}/SVMBaggingNFoldCrossValidator.h"
    #"${INCLUDE_DIR}/SVMDataMiner.h"
    "${INCLUDE_DIR}/SVMParams.h"
    "${INCLUDE_DIR}/template/SVMParams_template.h"
    "${INCLUDE_DIR}/SVMTrainer.h"
    "${INCLUDE_DIR}/template/SVMTrainer_template.h"
    "${INCLUDE_DIR}/ViewFeatureDetector.h"
    )

set( SRC_FILES
    ${SRC_DIR}/Classification
    ${SRC_DIR}/CrossValidator
    ${SRC_DIR}/CvModel
    ${SRC_DIR}/DataUtils
    ${SRC_DIR}/DecisionTreeRandomCrossValidator
    ${SRC_DIR}/DiscreteNaiveBayes
    ${SRC_DIR}/FeatureDetector
    ${SRC_DIR}/GaussianMAPEstimator
    #${SRC_DIR}/HOGModel
    ${SRC_DIR}/KMeans
    ${SRC_DIR}/KNearestClassifier
    ${SRC_DIR}/KNearestNFoldCrossValidator
    ${SRC_DIR}/KNearestRandomCrossValidator
    ${SRC_DIR}/MAPEstimator
    ${SRC_DIR}/Model
    ${SRC_DIR}/NaiveBayesRandomCrossValidator
    ${SRC_DIR}/NFoldCrossValidator
    ${SRC_DIR}/ObjectClassificationStatsManager
    ${SRC_DIR}/ObjectDetectionStatsManager
    ${SRC_DIR}/ObjectDetectionStatsManager_ViewInfo
    ${SRC_DIR}/PCA
    ${SRC_DIR}/PrecisionRecallFinder
    ${SRC_DIR}/RandomCrossValidator
    ${SRC_DIR}/RangePartsDetector
    ${SRC_DIR}/RealObjectSizeResponseSuppressor
    ${SRC_DIR}/StatsGenerator
    ${SRC_DIR}/SVMClassifier
    ${SRC_DIR}/SVMNFoldCrossValidator
    ${SRC_DIR}/SVMBaggingNFoldCrossValidator
    #${SRC_DIR}/SVMDataMiner
    #${SRC_DIR}/SVMModel
    ${SRC_DIR}/SVMParams
    #${SRC_DIR}/SVMViewExtractTrainer
    ${SRC_DIR}/ViewFeatureDetector
	)


# Require generate_export_header for .lib import libraries associated with Windows DLLs (if BUILD_SHARED_LIBS is true)
include(GenerateExportHeader)
add_library( ${PROJECT_NAME} ${SRC_FILES} ${INCLUDE_FILES})
generate_export_header( ${PROJECT_NAME}
    BASE_NAME ${PROJECT_NAME}
    EXPORT_MACRO_NAME ${PROJECT_NAME}_EXPORT
    EXPORT_FILE_NAME ${PROJECT_NAME}_Export.h
    STATIC_DEFINE ${PROJECT_NAME}_BUILT_AS_STATIC)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries( ${PROJECT_NAME} ${RFeatures_LIBRARY})
target_link_libraries( ${PROJECT_NAME} ${RLIB_LIBRARY})
target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries( ${PROJECT_NAME} ${LibLAS_LIBRARY})

set( CMAKE_INSTALL_PREFIX "${LOCALLIBS}/${PROJECT_NAME}")
message( STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
install( TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin"
                                 LIBRARY DESTINATION "lib"
                                 ARCHIVE DESTINATION "lib/static")

install( FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_Export.h" DESTINATION "include")
install( FILES ${INCLUDE_FILES} DESTINATION "include")
install( DIRECTORY "${INCLUDE_DIR}/template" DESTINATION "include")
# Copy across this library's Find cmake module to ${LOCALLIBS}/cmake
install( FILES "${PROJECT_SOURCE_DIR}/../CMakeModules/FindRLearning.cmake" DESTINATION "${LOCALLIBS}/cmake")

