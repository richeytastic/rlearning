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
#ifndef RLEARNING_OBJECT_DETECTION_STATS_MANAGER_VIEW_INFO_H
#define RLEARNING_OBJECT_DETECTION_STATS_MANAGER_VIEW_INFO_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/foreach.hpp>
#include <iostream>

namespace RLearning
{

struct ViewInfo
{
    ViewInfo(){}    // For operator>>
    ViewInfo( int i, const cv::Size& d, const std::string& s) : id(i), win(d), vstr(s) {}

    int id;             // View ID
    cv::Size win;       // View window size
    std::string vstr;   // string identifier
    std::vector<cv::Rect> grndTrth;    // Ground truth
    std::vector<cv::Rect> detBoxes;    // Detections
    std::vector<float> detConfs;       // Corresponding detection confidence scores
};  // end struct

std::ostream& operator<<( std::ostream& os, const ViewInfo& v);
std::istream& operator>>( std::istream& is, ViewInfo& v);

}   // end namespace
#endif

