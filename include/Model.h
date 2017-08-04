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
#ifndef RLEARNING_MODEL_H
#define RLEARNING_MODEL_H

#include <stdexcept>
#include <iostream>
using std::istream;
using std::ostream;
#include <string>
using std::string;
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>


namespace RLearning
{

class Model
{
public:
    // Read the model type from a given model file without reading in any
    // more data. Allows external classes to choose which child class of Model to
    // load based on the returned type identifier. In the event that the model type
    // could not be read, the string returned is INVALID_MODEL.
    static string readModelType( const string &modelFilename);

    static const string INVALID_MODEL;
    virtual ~Model(){}

    string getName() const { return name_;} // Identifier for this object model.
    cv::Size2f getActualSize() const { return objSz_;}   // Actual feature size (metres)
    string getFeatureType() const { return fvType_;}     // Feature type

    virtual string getModelType() const = 0;    // Model type

    // Predict the response or class from the given test example.
    virtual double predict( const cv::Mat&) = 0;

    class Exception : public std::runtime_error
    {
    public:
        Exception( const string &m) : std::runtime_error(m) {}
    };  // end class Exception

protected:
    // Create a model from knowledge of the object's real size (in metres).
    Model( const string &name, const cv::Size2f &objSz, const string &fvType);

    virtual void writeHeader( ostream &os) const;

    // Read in the header setting the name and size of this model and returning the model type.
    virtual string readHeader( istream &is) throw (Exception);

    // Reads in arbitrary tokens throwing exception if label is not matched
    // or expected token type does not match. Each new line of the input stream
    // should contain a single label and token in the format:
    // LABEL: TOKEN
    // The expected type of the token is given by the template parameter.
    template <typename T>
    static T readToken( istream &is, const string &label) throw (Model::Exception);

private:
    string name_;       // The name of this model (type of objects being detected)
    cv::Size2f objSz_;  // Object actual size in metres (all object models have a real size)
    string fvType_;     // The type of the feature vector

    Model();
};  // end class


#include "template/Model_template.h"

}   // end namespace

#endif



