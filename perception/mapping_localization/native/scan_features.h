#pragma once
#include "roll_support.h"
#include "tic_toc.h"
#include "FeatureTypes.h"
#define ROS_WARN(...) do{}while(0)
#define ROS_BREAK() throw std::runtime_error("Unsupported LiDAR scan geometry")
class Features:public ParamServer {
    pcl::VoxelGrid<PointType> downSizeFilter;
    vector<smoothness_t> cloudSmoothness;
    vector<float> cloudCurvature;
    vector<int> cloudNeighborPicked,cloudLabel;
#include "FeatureFilter.inc"
public:
    explicit Features(const YAML::Node& config):ParamServer(config) {
        downSizeFilter.setLeafSize(odometrySurfLeafSize,odometrySurfLeafSize,odometrySurfLeafSize);
        size_t n=N_SCAN*Horizon_SCAN*10;cloudCurvature.resize(n);cloudNeighborPicked.resize(n);cloudLabel.resize(n);cloudSmoothness.resize(n);
    }
    void compute(pcl::PointCloud<PointType>::Ptr lidarCloudIn,pcl::PointCloud<PointType>::Ptr& corners,pcl::PointCloud<PointType>::Ptr& surfaces) {
        TicToc t_whole;vector<int> scanStartInd(N_SCAN,0),scanEndInd(N_SCAN,0);
#include "FeatureMath.inc"
        corners=cornerCloud;surfaces=surfaceCloud;
    }
};
