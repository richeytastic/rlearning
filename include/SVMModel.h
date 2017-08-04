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
#ifndef RLEARNING_SVM_MODEL
#define RLEARNING_SVM_MODEL

#include "Model.h"
using RLearning::Model;
#include "SVMClassifier.h"
using RLearning::SVMClassifier;


namespace RLearning
{

class SVMModel : public Model
{
public:
    static const string Type;   // "SVM"
    typedef boost::shared_ptr<SVMModel> Ptr;

    SVMModel();  // For use with operator>>
    SVMModel( const string &name,      // Name of the object
              const cv::Size2f &objSz, // Real size of the object (as trained)
              const SVMClassifier::Ptr &svmc); // The trained classifier
    virtual ~SVMModel() {}

    virtual cv::Size getModelDims( int *nbins) const;
    virtual string getModelType() const;

    virtual double predict( const cv::Mat &z);

protected:
    virtual void writeHeader( ostream&) const;
    virtual void writeBody( ostream&) const;    // Writes out SVM byte data

    virtual string readHeader( istream&) throw (Model::Exception);
    virtual void readBody( istream&);   // Reads in SVM byte data

    friend ostream &operator<<( ostream&, const SVMModel&);
    friend istream &operator>>( istream&, SVMModel&);

private:
    SVMClassifier::Ptr svmc;    // The SVM classifier
};  // end class


ostream &operator<<( ostream&, const SVMModel&);
istream &operator>>( istream&, SVMModel&);

}   // end namespace

#endif
