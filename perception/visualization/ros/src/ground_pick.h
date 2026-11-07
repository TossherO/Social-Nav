#pragma once
#include <cmath>
#include <limits>
#include <vector>

namespace social_nav_rviz {
struct Point { double x, y, z; };

inline bool plane_hit(const Point& origin, const Point& direction, double z, Point& hit) {
  if (!std::isfinite(z) || std::abs(direction.z) < 1e-9) return false;
  const double t = (z-origin.z)/direction.z;
  if (!std::isfinite(t) || t < 0.) return false;
  hit = {origin.x+t*direction.x, origin.y+t*direction.y, z};
  return std::isfinite(hit.x) && std::isfinite(hit.y);
}

// Each known grid cell is a horizontal tile at its measured display height.
// Select the closest ray intersection; never extend a floor across a hole.
inline bool ground_hit(const std::vector<Point>& cells, double resolution,
                       const Point& origin, const Point& direction, Point& hit) {
  if (!std::isfinite(resolution) || resolution <= 0.) return false;
  double nearest = std::numeric_limits<double>::infinity();
  for (const auto& cell : cells) {
    Point candidate;
    if (!plane_hit(origin,direction,cell.z,candidate)) continue;
    if (std::abs(candidate.x-cell.x) > resolution*.5 ||
        std::abs(candidate.y-cell.y) > resolution*.5) continue;
    const double t = (cell.z-origin.z)/direction.z;
    if (t < nearest) { nearest=t; hit=candidate; }
  }
  return std::isfinite(nearest);
}
} // namespace social_nav_rviz
