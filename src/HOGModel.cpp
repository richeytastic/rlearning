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

#include "HOGModel.h"
using RLearning::HOGModel;


const string HOGModel::Type = "HOG";


HOGModel::HOGModel()
{ }   // end ctor


HOGModel::HOGModel( const string &nm, const cv::Size2f &ms,
                    const SVMClassifier::Ptr &s, bool dd, int cs)
    : ProHOGModel( nm, ms, s, dd), cellPxls(cs)
{ }   // end ctor


string HOGModel::getModelType() const
{
    return HOGModel::Type;
}   // end getModelType


void HOGModel::writeHeader( ostream &s) const
{
    ProHOGModel::writeHeader(s);
    s << "CELL_PIXELS: " << cellPxls << endl;
}   // end writeHeader


string HOGModel::readHeader( istream &s) throw (ObjModel::Exception)
{
    string typ = ProHOGModel::readHeader(s);
    cellPxls = ObjModel::readToken<int>( s, "CELL_PIXELS:");    // may throw
    return typ;
}   // end readHeader


ostream& RLearning::operator<<( ostream &s, const HOGModel &mod)
{
    mod.writeHeader(s);
    mod.writeBody(s);  // Not overridden here so calls ProHOGModel::writeBody
    return s;
}   // end operator<<


istream& RLearning::operator>>( istream &s, HOGModel &mod)
{
    mod.readHeader(s);
    mod.readBody(s);   // Not overridden here so calls ProHOGModel::readBody
    return s;
}   // end operator>>
