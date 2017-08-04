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

#include <sstream>
using std::istringstream;
using std::ostringstream;
#include <ios>


static void failAndThrow( istream &is, const string &msg) throw (Model::Exception)
{
    is.setstate( std::ios::failbit);
    throw Model::Exception( msg);
}   // end failAndThrow


template <typename T>
T Model::readToken( istream &is, const string &lab) throw (Model::Exception)
{
    string inLine;
    std::getline( is, inLine);
    istringstream iss(inLine);
    string inLab;
    iss >> inLab;
    string lb = lab;
    boost::algorithm::to_lower(lb); // Ensure provided label is lower case
    boost::algorithm::to_lower(inLab);  // Ensure read label is lower case
    if ( inLab.compare(lb) != 0)
    {
        ostringstream oss;
        oss << "Tried to read token " << lab
            << " into Model instance but found token " << inLab << " instead!";
        failAndThrow( is, oss.str());
    }   // end if

    T tok;

    try
    {
        iss >> tok;
    }   // end try
    catch ( const std::exception &e)
    {
        ostringstream oss;
        oss << "Tried and failed to read token " << lab << " into Model instance!";
        failAndThrow( is, oss.str());
    }   // end catch

    return tok;
}   // end readToken
