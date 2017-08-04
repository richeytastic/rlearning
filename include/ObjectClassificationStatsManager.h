/************************************************************************
 * Copyright (C) 2017 Richard Palmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

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
