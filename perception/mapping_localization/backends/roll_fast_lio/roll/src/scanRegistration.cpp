// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "utility.h"
#include "roll/cloud_info.h"

#include "FeatureTypes.h"
class scanRegistration : public ParamServer
{

private:
    ros::NodeHandle nh;
    ros::Subscriber subLidarCloudInfo;

    ros::Publisher pubRawPoints;
    ros::Publisher pubProjPoints;
    
    ros::Publisher pubLidarCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;
    ros::Publisher pubSharpCornerPoints;
    ros::Publisher pubFlatSurfacePoints;

    pcl::PointCloud<PointType>::Ptr surfaceCloud2;

    ros::Publisher pubSurfacePoints2;

    pcl::VoxelGrid<PointType> downSizeFilter;

    vector<smoothness_t> cloudSmoothness;
    vector<float> cloudCurvature;
    vector<int> cloudNeighborPicked;
    vector<int> cloudLabel;

    // vector<pcl::PointCloud<PointType>> lidarCloudScans;

public:
    scanRegistration()
    {
        subLidarCloudInfo = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 64, &scanRegistration::lidarCloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubLidarCloudInfo = nh.advertise<roll::cloud_info> ("/roll/feature/cloud_info", 1);
        pubRawPoints = nh.advertise<sensor_msgs::PointCloud2>("/roll/feature/cloud_raw", 1);
        pubProjPoints = nh.advertise<sensor_msgs::PointCloud2>("/roll/feature/cloud_projected", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("/roll/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("/roll/feature/cloud_surface", 1);
        pubSurfacePoints2 = nh.advertise<sensor_msgs::PointCloud2>("/roll/feature/cloud_surface2", 1);
        pubFlatSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("/roll/feature/cloud_surface_flat", 1);
        pubSharpCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("/roll/feature/cloud_corner_sharp", 1);
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        // to avoid overflow (sometimes points in one frame can be a lot)
        cloudCurvature.resize(N_SCAN*Horizon_SCAN*10);
        cloudNeighborPicked.resize(N_SCAN*Horizon_SCAN*10);
        cloudLabel.resize(N_SCAN*Horizon_SCAN*10);
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN*10);
    }

    // very important in outdoor SLAM
    // in nclt 20120202 loc run, it reduces reprojection error
    #include "FeatureFilter.inc"
    void lidarCloudHandler(const sensor_msgs::PointCloud2ConstPtr &lidarCloudMsg)
    {

        roll::cloud_info cloudInfo;

        TicToc t_whole;
        TicToc t_prepare;
        std::vector<int> scanStartInd(N_SCAN, 0);
        std::vector<int> scanEndInd(N_SCAN, 0);

        ros::Time lidarMsgStamp = lidarCloudMsg->header.stamp;
        pcl::PointCloud<PointType>::Ptr lidarCloudIn(new pcl::PointCloud<PointType>());
        
        pcl::fromROSMsg(*lidarCloudMsg, *lidarCloudIn);
            
        publishCloud(&pubRawPoints,  lidarCloudIn,  lidarMsgStamp, lidarFrame);
    
        cloudInfo.cloud_raw = *lidarCloudMsg;
        #include "FeatureMath.inc"
        publishCloud(&pubProjPoints,  lidarCloud,  lidarMsgStamp, lidarFrame);
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  lidarMsgStamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, lidarMsgStamp, lidarFrame);
        cloudInfo.cloud_corner_sharp  = publishCloud(&pubSharpCornerPoints,  cornerCloudSharp,  lidarMsgStamp, lidarFrame);
        cloudInfo.cloud_surface_flat = publishCloud(&pubFlatSurfacePoints, surfaceCloudFlat, lidarMsgStamp, lidarFrame);
        cloudInfo.header = lidarCloudMsg->header;
        // cout<<"finish"<<endl;
        pubLidarCloudInfo.publish(cloudInfo);

        // printf("scan registration time %f ms *************\n", t_whole.toc()); // ~ 40 ms
        if(t_whole.toc() > 100)
            ROS_WARN("scan registration process over 100ms");
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");

    scanRegistration SR;
    ROS_INFO("\033[1;32m----> Lidar Scan Registration Started.\033[0m");
    ros::spin();

    return 0;
}
