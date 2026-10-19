#pragma once
// Map format and I/O are independent of the estimator and its worker threads.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

namespace roll_storage {
namespace fs = std::filesystem;

template<class PosePoint>
void load(const std::string& directory, pcl::PointCloud<PosePoint>& poses,
          pcl::PointCloud<pcl::PointXYZI>& positions,
          std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& corners,
          std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& surfaces,
          std::vector<int>& indoor) {
    std::ifstream input(fs::path(directory) / "poses.txt");
    if (!input) throw std::runtime_error("Cannot open map poses: " + directory);
    std::string line;
    size_t lineNumber = 0, columns = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::istringstream row(line);
        std::vector<double> values;
        double value;
        while (row >> value) values.push_back(value);
        if (!row.eof() || (values.size() != 7 && values.size() != 8 && values.size() != 9) ||
            (columns && values.size() != columns) ||
            std::any_of(values.begin(), values.end(), [](double x){ return !std::isfinite(x); }))
            throw std::runtime_error("Invalid poses.txt row " + std::to_string(lineNumber));
        columns = values.size();
        size_t offset = columns == 9 ? 1 : 0;
        if (values[offset + 6] != static_cast<double>(poses.size()))
            throw std::runtime_error("Non-contiguous map index at row " + std::to_string(lineNumber));
        PosePoint pose{};
        pose.x = values[offset]; pose.y = values[offset + 1]; pose.z = values[offset + 2];
        pose.roll = values[offset + 3]; pose.pitch = values[offset + 4]; pose.yaw = values[offset + 5];
        pose.intensity = poses.size(); pose.time = columns == 9 ? values[0] : 0;
        pcl::PointXYZI position;
        position.x = pose.x; position.y = pose.y; position.z = pose.z; position.intensity = pose.intensity;
        auto corner = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
        auto surface = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
        const auto index = std::to_string(poses.size());
        for (auto entry : {std::make_pair("corner", corner), std::make_pair("surf", surface)}) {
            auto file = fs::path(directory) / (entry.first + index + ".pcd");
            if (pcl::io::loadPCDFile(file.string(), *entry.second) < 0)
                throw std::runtime_error("Cannot read map cloud: " + file.string());
            for (const auto& point : *entry.second)
                if (!pcl::isFinite(point)) throw std::runtime_error("Non-finite map cloud: " + file.string());
        }
        if (surface->empty()) throw std::runtime_error("Empty surface keyframe " + index);
        poses.push_back(pose); positions.push_back(position);
        corners.push_back(corner); surfaces.push_back(surface);
        indoor.push_back(columns >= 8 ? static_cast<int>(values.back()) : 1);
    }
    if (poses.empty()) throw std::runtime_error("Map has no keyframes: " + directory);
}

inline void checkedWrite(const fs::path& file, const pcl::PointCloud<pcl::PointXYZI>& cloud) {
    if (pcl::io::savePCDFileBinary(file.string(), cloud) != 0)
        throw std::runtime_error("Failed to write map cloud: " + file.string());
}

template<class PosePoint>
void save(const fs::path& mapDirectory, const pcl::PointCloud<PosePoint>& poses,
          const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& corners,
          const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& surfaces,
          const std::vector<int>& indoor, double resolution, const std::string& frame) {
    if (poses.empty() || poses.size() != corners.size() || poses.size() != surfaces.size() || poses.size() != indoor.size())
        throw std::runtime_error("Inconsistent or empty map snapshot");
    if (!std::isfinite(resolution) || resolution <= 0) throw std::runtime_error("Invalid export resolution");
    const auto keyframes = mapDirectory / "keyframes";
    if (fs::exists(keyframes) && !fs::is_empty(keyframes))
        throw std::runtime_error("Refusing to overwrite saved keyframes: " + keyframes.string());
    const fs::path staging = mapDirectory.string() + ".saving";
    if (fs::exists(staging)) throw std::runtime_error("Incomplete save exists: " + staging.string());
    fs::create_directories(staging / "keyframes");
    std::ofstream rows(staging / "keyframes" / "poses.txt");
    rows << std::setprecision(12);
    pcl::PointCloud<pcl::PointXYZI> global;
    for (size_t i = 0; i < poses.size(); ++i) {
        const auto& pose = poses[i];
        checkedWrite(staging / "keyframes" / ("corner" + std::to_string(i) + ".pcd"), *corners[i]);
        checkedWrite(staging / "keyframes" / ("surf" + std::to_string(i) + ".pcd"), *surfaces[i]);
        rows << pose.time << ' ' << pose.x << ' ' << pose.y << ' ' << pose.z << ' '
             << pose.roll << ' ' << pose.pitch << ' ' << pose.yaw << ' ' << i << ' ' << indoor[i] << '\n';
        const Eigen::Affine3f transform = pcl::getTransformation(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw);
        for (const auto& cloud : {corners[i], surfaces[i]}) {
            pcl::PointCloud<pcl::PointXYZI> world;
            pcl::transformPointCloud(*cloud, world, transform);
            global += world;
        }
    }
    rows.flush();
    if (!rows) throw std::runtime_error("Failed to write poses.txt");
    rows.close();
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(resolution, resolution, resolution);
    voxel.setInputCloud(global.makeShared());
    pcl::PointCloud<pcl::PointXYZI> display;
    voxel.filter(display);
    checkedWrite(staging / "GlobalMap.pcd", display);
    std::ofstream manifest(staging / "manifest.yaml");
    manifest << "format: roll_keyframes_v1\nframe: " << frame << "\nunits: metres_radians\nkeyframes: " << poses.size()
             << "\nvisualization_resolution: " << resolution << "\n";
    manifest.flush();
    if (!manifest) throw std::runtime_error("Failed to write map manifest");
    manifest.close();
    std::ofstream complete(staging / "COMPLETE");
    complete << "roll_keyframes_v1\n";
    complete.close();
    if (!complete) throw std::runtime_error("Failed to mark completed map");
    // Publish the entire completed bundle in one rename on the same filesystem.
    for (const auto& entry : fs::directory_iterator(mapDirectory))
        if (entry.path() != keyframes) throw std::runtime_error("Unexpected file in new map output: " + entry.path().string());
    if (fs::exists(keyframes)) fs::remove(keyframes);
    fs::remove(mapDirectory);
    fs::rename(staging, mapDirectory);
}
} // namespace roll_storage
