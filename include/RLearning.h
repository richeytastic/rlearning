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

#include "Classification.h"
#include "CrossValidator.h"
#include "CvModel.h"
#include "DataUtils.h"
#include "DecisionTreeRandomCrossValidator.h"
#include "DiscreteNaiveBayes.h"
#include "FeatureDetector.h"
#include "GaussianMAPEstimator.h"
#include "KernelCache.h"
#include "KMeans.h"
#include "KNearestClassifier.h"
#include "KNearestNFoldCrossValidator.h"
#include "KNearestRandomCrossValidator.h"
#include "MAPEstimator.h"
#include "Model.h"
#include "NaiveBayesRandomCrossValidator.h""
#include "NFoldCrossValidator.h"
#include "PCA.h"
#include "PrecisionRecallFinder.h"
#include "RandomCrossValidator.h"
#include "ROCFinder.h"
#include "SVMClassifier.h"
#include "SVMNFoldCrossValidator.h"
#include "SVMDataMiner.h"
#include "SVMParams.h"
#include "SVMTrainer.h"
#include "ViewFeatureDetector.h"
