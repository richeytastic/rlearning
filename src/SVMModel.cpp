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

#include "SVMModel.h"
using RLearning::SVMModel;
#include <boost/algorithm/string.hpp>
#include <sstream>
using std::ostringstream;
#include <cassert>
#include <iostream>
using std::endl;

const string SVMModel::Type = "SVM";


SVMModel::SVMModel()
    : Model( "", cv::Size2f(0,0))
{
}   // end ctor


SVMModel::SVMModel( const string &nm, const cv::Size2f &ms, const SVMClassifier::Ptr &s)
    : Model( nm, ms), svmc(s)
{ }   // end ctor


cv::Size SVMModel::getModelDims( int *nbins) const
{
    return svmc->getModelDims( nbins);
}   // end getModelDims


string SVMModel::getModelType() const
{
    return SVMModel::Type;
}   // end getModelType


double SVMModel::predict( const cv::Mat &z) 
{
    return svmc->predict(z);
}   // end predict


void SVMModel::writeHeader( ostream &s) const
{
    Model::writeHeader(s);
}   // end writeHeader


void SVMModel::writeBody( ostream &s) const
{
    s << *svmc << endl;
}   // end writeBody


string SVMModel::readHeader( istream &s) throw (Model::Exception)
{
    string typ = Model::readHeader(s);
    //string dirDepStr = Model::readToken<string>( s, "CONTRAST_DIRECTION:");  // may throw
    //dirDep = false;
    //boost::algorithm::to_lower(dirDepStr);
    //if ( dirDepStr.compare("yes") == 0)
    //    dirDep = true;
    return typ;
}   // end readHeader


void SVMModel::readBody( istream &s)
{
    svmc = SVMClassifier::create();
    s >> *svmc;
    string ln;
    std::getline(s,ln);
}   // end readBody


ostream& RLearning::operator<<( ostream &s, const SVMModel &mod)
{
    mod.writeHeader(s);
    mod.writeBody(s);
    return s;
}   // end operator<<


istream& RLearning::operator>>( istream &s, SVMModel &mod)
{
    mod.readHeader(s);
    mod.readBody(s);
    return s;
}   // end operator>>

