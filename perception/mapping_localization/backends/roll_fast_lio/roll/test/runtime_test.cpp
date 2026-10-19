#include <gtest/gtest.h>
#include "FixedMapSubmap.h"

struct TestPose {
    float x = 0, y = 0, z = 0, intensity = 0, roll = 0, pitch = 0, yaw = 0;
};

TEST(FixedMapSubmap, ReplacedRequestCannotPublishOldGenerationAndRecoveryIsFresh) {
    using Cloud = pcl::PointCloud<PointType>;
    pcl::PointCloud<TestPose> poses;
    std::vector<Cloud::Ptr> corners, surfaces;
    for (int id = 0; id < 2; ++id) {
        TestPose pose; pose.x = id*10; pose.intensity = id; poses.push_back(pose);
        Cloud::Ptr cloud(new Cloud);
        for (int i = 0; i < 6; ++i) {
            PointType p; p.x = i*.1f; p.y = 0; p.z = 0; p.intensity = i;
            cloud->push_back(p);
        }
        corners.push_back(cloud); surfaces.push_back(cloud);
    }
    FixedMapSubmap<TestPose> worker(poses, corners, surfaces, 1, .1f, .05f, .05f, 1, 4096);
    worker.request(Eigen::Vector3f(0,0,0), 1);
    worker.request(Eigen::Vector3f(10,0,0), 2);
    auto next = worker.take(2, true);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->epoch, 2);
    ASSERT_EQ(next->ids.size(), 1);
    EXPECT_EQ(next->ids[0], 1);
    EXPECT_NEAR(next->surfaceDS->front().x, 10, 1e-5);
    EXPECT_FALSE(worker.take(1, false));

    worker.request(Eigen::Vector3f(1000,0,0), 3);
    auto missing = worker.take(3, true);
    EXPECT_TRUE(missing->ids.empty());
    EXPECT_TRUE(missing->surfaceDS->empty());
    worker.request(Eigen::Vector3f(0,0,0), 4);
    auto recovered = worker.take(4, true);
    ASSERT_EQ(recovered->ids.size(), 1);
    EXPECT_EQ(recovered->ids[0], 0);
    EXPECT_NEAR(recovered->surfaceDS->front().x, 0, 1e-5);
    EXPECT_EQ(surfaces[0]->front().x, 0); // Frozen source geometry was never transformed in place.
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
