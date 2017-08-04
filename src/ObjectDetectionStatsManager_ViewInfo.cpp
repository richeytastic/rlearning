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

#include "ObjectDetectionStatsManager_ViewInfo.h"
using RLearning::ViewInfo;


std::ostream& RLearning::operator<<( std::ostream& os, const ViewInfo& v)
{
    os << v.id << " " << v.vstr << " " << v.win.width << " " << v.win.height;

    os << " " << v.grndTrth.size(); // Add the ground truth boxes
    BOOST_FOREACH ( const cv::Rect& gt, v.grndTrth)
        os << " " << gt.x << " " << gt.y << " " << gt.width << " " << gt.height;

    os << " " << v.detBoxes.size(); // Add the detection scores and boxes
    assert( v.detBoxes.size() == v.detConfs.size());
    for ( int i = 0; i < v.detBoxes.size(); ++i)
    {
        const cv::Rect& dbox = v.detBoxes[i];
        os << " " << v.detConfs[i] << " " << dbox.x << " " << dbox.y << " " << dbox.width << " " << dbox.height;
    }   // end for

    return os;
}   // end operator<<


std::istream& RLearning::operator>>( std::istream& is, ViewInfo& v)
{
    is >> v.id >> v.vstr >> v.win.width >> v.win.height;
    int numgt;
    is >> numgt;
    v.grndTrth.resize(numgt);
    for ( int i = 0; i < numgt; ++i)
    {
        cv::Rect& gt = v.grndTrth[i];
        is >> gt.x >> gt.y >> gt.width >> gt.height;
    }   // end for

    int numdets;
    is >> numdets;
    v.detConfs.resize(numdets);
    v.detBoxes.resize(numdets);
    for ( int i = 0; i < numdets; ++i)
    {
        cv::Rect& dbox = v.detBoxes[i];
        is >> v.detConfs[i] >> dbox.x >> dbox.y >> dbox.width >> dbox.height;
    }   // end for

    return is;
}   // end operator>>
