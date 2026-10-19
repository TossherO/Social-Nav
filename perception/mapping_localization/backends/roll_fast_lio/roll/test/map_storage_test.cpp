#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/common/point_tests.h>
#include "MapStorage.h"

struct TestPose {
    float x=0, y=0, z=0, roll=0, pitch=0, yaw=0, intensity=0;
    double time=0;
};

class MapStorageTest : public ::testing::Test {
protected:
    std::filesystem::path root;
    pcl::PointCloud<TestPose> poses;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> corners, surfaces;
    std::vector<int> indoor;
    void SetUp() override {
        char path[]="/tmp/roll-map-test-XXXXXX";
        root=mkdtemp(path);
        std::filesystem::create_directories(root/"map/keyframes");
        for (int i=0; i<2; ++i) {
            TestPose pose; pose.x=i; pose.time=100+i; pose.intensity=i;
            poses.push_back(pose);
            auto cloud=pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointXYZI point; point.x=1; point.y=2; point.z=3; point.intensity=4;
            cloud->push_back(point);
            corners.push_back(cloud); surfaces.push_back(cloud); indoor.push_back(1);
        }
    }
    void TearDown() override { std::filesystem::remove_all(root); }
    void save() { roll_storage::save(root/"map", poses,corners,surfaces,indoor,0.1,"map"); }
    void load() {
        pcl::PointCloud<TestPose> loaded;
        pcl::PointCloud<pcl::PointXYZI> positions;
        decltype(corners) c,s;
        std::vector<int> i;
        roll_storage::load((root/"map/keyframes").string(),loaded,positions,c,s,i);
        ASSERT_EQ(loaded.size(),2u);
        EXPECT_FLOAT_EQ(loaded[1].x,1);
        EXPECT_DOUBLE_EQ(loaded[1].time,101);
    }
};

TEST_F(MapStorageTest, RoundTripAndNoOverwrite) {
    save(); load();
    EXPECT_TRUE(std::filesystem::exists(root/"map/COMPLETE"));
    EXPECT_FALSE(std::filesystem::exists(root/"map.saving"));
    EXPECT_THROW(save(),std::runtime_error);
    load();
}
TEST_F(MapStorageTest, FinalRowWithoutNewlineIsRetained) {
    save();
    auto path=root/"map/keyframes/poses.txt";
    std::filesystem::resize_file(path,std::filesystem::file_size(path)-1);
    load();
}
TEST_F(MapStorageTest, MissingCloudIsRejected) {
    save();
    std::filesystem::remove(root/"map/keyframes/surf1.pcd");
    EXPECT_THROW(load(),std::runtime_error);
}
TEST_F(MapStorageTest, InvalidPoseIsRejected) {
    save();
    std::ofstream(root/"map/keyframes/poses.txt") << "100 0 0 0 0 0 0 9 1\n";
    EXPECT_THROW(load(),std::runtime_error);
}
TEST_F(MapStorageTest, IncompleteSaveIsNotReplaced) {
    std::filesystem::create_directory(root/"map.saving");
    EXPECT_THROW(save(),std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(root/"map/COMPLETE"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
