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
