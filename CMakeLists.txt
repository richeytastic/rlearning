cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
project( rLearning)

set(WITH_RFEATURES TRUE)
include( "cmake/LibCommon.cmake")

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set( SRC_DIR "${PROJECT_SOURCE_DIR}/src")
include_directories( ${INCLUDE_DIR})

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

add_library( ${PROJECT_NAME} ${SRC_FILES} ${INCLUDE_FILES})
include( "cmake/LibInstall.cmake")
