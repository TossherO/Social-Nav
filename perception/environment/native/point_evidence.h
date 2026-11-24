#pragma once

#include <cstdint>

enum PointEvidence : uint8_t {
    OrdinaryHit = 0,
    MovablePersonHit = 1,
    UnconfirmedHit = 2
};
