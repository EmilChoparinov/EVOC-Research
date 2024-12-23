#ifndef _RANDOMNESS_HEADER_H
#define _RANDOMNESS_HEADER_H

///////////////////////////////////////////////////////////////////////////////////////////
//    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
//
//    Contact info:
//
//    Peter Chervenski < spookey@abv.bg >
//    Shane Ryan < shane.mcdonald.ryan@gmail.com >
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Random.h
// Description: Declarations for a class dealing with random numbers.
///////////////////////////////////////////////////////////////////////////////

#include <random>
#include <vector>
#include <limits>
#include <stdexcept>
#include <cereal/cereal.hpp>
#include <sstream>

namespace NEAT
{

    class RNG
    {
        std::mt19937 rng;

    public:
        RNG();

        // Seeds the random number generator with this value
        void Seed(long seed);

        // Seeds the random number generator with time
        void TimeSeed();

        // Returns randomly either 1 or -1
        int RandPosNeg();

        // Returns a random integer between X and Y
        int RandInt(int x, int y);

        // Returns a random number from a uniform distribution in the range of [0 .. 1]
        double RandFloat();

        // Returns a random number from a uniform distribution in the range of [-1 .. 1]
        double RandFloatSigned();

        // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
        double RandGaussSigned();

        // Returns an index given a vector of probabilities
        int Roulette(std::vector<double> &a_probs);

        template <class Archive>
        void save(Archive &ar) const
        {
            std::stringstream ss;
            ss << rng;
            std::string serialized = ss.str();
            ar &serialized;
        }

        template <class Archive>
        void load(Archive &ar)
        {
            std::string serialized;
            ar &serialized;
            std::stringstream ss;
            ss << serialized;
            ss >> rng;
        }
    };

} // namespace NEAT

#endif
