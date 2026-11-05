#pragma once
struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

using std::atan2;
using std::cos;
using std::sin;

double scanPeriod = 0.1;
