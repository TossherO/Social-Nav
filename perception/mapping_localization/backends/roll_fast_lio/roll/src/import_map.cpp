// Validate legacy assets with the estimator's map reader, then reference them without copying clouds.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <pcl/common/point_tests.h>
#include "MapStorage.h"

struct ImportPose {
    float x=0, y=0, z=0, roll=0, pitch=0, yaw=0, intensity=0;
    double time=0;
};

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc != 3) {
        std::cerr << "Usage: roll_import_map SOURCE_MAP NEW_REFERENCE_MAP\n";
        return 2;
    }
    const fs::path output = fs::absolute(argv[2]);
    const fs::path staging = output.string() + ".importing";
    try {
        const fs::path source = fs::canonical(argv[1]);
        if (fs::exists(output) || fs::exists(staging))
            throw std::runtime_error("Refusing existing map output or incomplete import");
        pcl::PointCloud<ImportPose> poses;
        pcl::PointCloud<pcl::PointXYZI> positions;
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> corners, surfaces;
        std::vector<int> indoor;
        roll_storage::load((source / "keyframes").string(), poses, positions, corners, surfaces, indoor);
        // These historic maps provide acquisition time; keep the original nine-column rows unchanged.
        std::ifstream rows(source / "keyframes/poses.txt");
        std::string line;
        std::getline(rows, line);
        std::istringstream first(line);
        double value;
        int columns = 0;
        while (first >> value) ++columns;
        if (columns != 9) throw std::runtime_error("This importer requires timestamped nine-column poses");
        fs::create_directories(staging);
        fs::create_directory_symlink(fs::relative(source / "keyframes", staging), staging / "keyframes");
        if (fs::is_regular_file(source / "GlobalMap.pcd"))
            fs::create_symlink(fs::relative(source / "GlobalMap.pcd", staging), staging / "GlobalMap.pcd");
        std::ofstream manifest(staging / "manifest.yaml");
        manifest << "format: roll_keyframes_v1\nframe: map\nunits: metres_radians\nkeyframes: "
                 << poses.size() << "\nreference_bundle: true\ngeometry_accuracy_verified: false\n"
                 << "source: " << std::quoted(source.string()) << "\n";
        manifest.close();
        if (!manifest) throw std::runtime_error("Failed to write import source record");
        std::ofstream complete(staging / "COMPLETE");
        complete << "roll_keyframes_v1\n";
        complete.close();
        if (!complete) throw std::runtime_error("Failed to mark completed import");
        fs::rename(staging, output);
        std::cout << "Imported " << poses.size() << " keyframes; clouds reference the unchanged source map.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Map import failed: " << error.what() << '\n';
        return 1;
    }
}
