// Thin C ABI over official Noetic global_planner search and traceback.
// No ROS node/master, cell clearing, plugin loading, or motion publisher.
#include <global_planner/astar.h>
#include <global_planner/dijkstra.h>
#include <global_planner/grid_path.h>
#include <global_planner/quadratic_calculator.h>
#include <costmap_2d/costmap_2d.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

extern "C" int social_nav_plan(const unsigned char* input, int width, int height,
                              int sx, int sy, int gx, int gy, int astar,
                              float* output, int capacity, double* elapsed_ms) noexcept {
  const auto begin = std::chrono::steady_clock::now();
  try {
    if (!input || !output || !elapsed_ms || width < 1 || height < 1 ||
        static_cast<long long>(width + 4) * (height + 4) > 4100000 ||
        sx < 0 || sy < 0 || gx < 0 || gy < 0 || sx >= width || gx >= width || sy >= height || gy >= height)
      return -1;
    if (input[sx + sy * width] >= 253 || input[gx + gy * width] >= 253) return -1;
    // Lethal guard cells prevent row wrap and unchecked neighbour accesses.
    const int nx = width + 4, ny = height + 4;
    costmap_2d::Costmap2D grid(nx, ny, 1., 0., 0., 254);
    for (int y = 0; y < height; ++y)
      std::copy(input + y * width, input + (y + 1) * width, grid.getCharMap() + (y + 2) * nx + 2);
    sx += 2; sy += 2; gx += 2; gy += 2;
    global_planner::QuadraticCalculator calculator(nx, ny);
    std::unique_ptr<global_planner::Expander> search;
    if (astar) search.reset(new global_planner::AStarExpansion(&calculator, nx, ny));
    else search.reset(new global_planner::DijkstraExpansion(&calculator, nx, ny));
    search->setSize(nx, ny);
    search->setHasUnknown(false);
    search->setNeutralCost(50);
    search->setFactor(1.0);  // Same cost units as AStarExpansion.
    std::vector<float> potential(nx * ny);
    const bool found = search->calculatePotentials(grid.getCharMap(), sx, sy, gx, gy, nx * ny * 2, potential.data());
    std::vector<std::pair<float, float>> path;
    global_planner::GridPath traceback(&calculator);
    traceback.setSize(nx, ny);
    const bool traced = found && traceback.getPath(potential.data(), sx, sy, gx, gy, path);
    *elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (!traced) return 0;
    if (path.size() > static_cast<size_t>(capacity)) return -2;
    std::reverse(path.begin(), path.end());
    for (size_t i = 0; i < path.size(); ++i) {
      output[2 * i] = path[i].first - 2;
      output[2 * i + 1] = path[i].second - 2;
    }
    return static_cast<int>(path.size());
  } catch (...) { return -3; }
}
