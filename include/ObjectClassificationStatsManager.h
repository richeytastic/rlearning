#pragma once
#ifndef RLEARNING_OBJECT_CLASSIFICATION_STATS_MANAGER_H
#define RLEARNING_OBJECT_CLASSIFICATION_STATS_MANAGER_H
#include <ObjectDetectionStatsManager.h>


namespace RLearning
{

class ObjectClassificationStatsManager : public ObjectDetectionStatsManager
{
public:
    ObjectClassificationStatsManager();

    bool addPositive( int id);
    bool addNegative( int id);
    bool classifyPositive( int id, float confidence);
    bool classifyNegative( int id, float confidence);

    // Calculate the TP, FP, TN, FP over all classifications having confidence scores >= minConfidence.
    virtual void calcStats( double& tp, double& fn, double& tn, double& fp, double minConfidence=0) const;

    void clearClassifications(); // Reset to do a new round
};  // end class


}   // end namespace
#endif
