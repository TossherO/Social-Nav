#pragma once
#ifndef _UTILITY_H_
#define _UTILITY_H_

#include"tic_toc.h"
#include <ceres/ceres.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include<pcl/visualization/pcl_visualizer.h>


#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include<cassert>
#include <utility>
#include <cstdlib>
#include <memory>
#include <filesystem>
#include <condition_variable>
#include <stdexcept>
#include <numeric>

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER, LIVOX};

class ParamServer
{
public:
    double globalMatchingRate = 10;
    int gnss_init_num = 10;

    vector<float>  array_imu_to_body;
    vector<float>  array_lidar_to_body;
    vector<float>  array_gps_to_body;
    vector<float> initialGuess;
    vector<float> gps_variance;
    vector<float> odom_variance;
    vector<float> loop_variance;

    bool bodyCentered;
    bool edgeEnabled;
    bool overlapFilterEnabled;
    bool gtAvailable;

    bool bashRun = false;
    string map_dir = "";
    string log_dir = "";

    bool debugMode = false;
    bool useGPS = false;

    // gps 
    double alti0;
    double lati0;
    double longi0;

    int optIteration;
    float degeneracyEigenThreshold;
    ros::NodeHandle nh;
    
    std::string robot_id;

    //Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;
    string gtTopic;
    string odometryTopic;

    //Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    bool savePose;
    bool saveKeyframeMap;
    
    bool mapUpdateEnabled;
    bool temporaryMappingEnabled;
    bool latestScanOnly;
    bool enableSecondaryFusion;
    int visualizationPathLimit;
    bool useKeyframeCache;
    bool gravityAlignedMap;
    int keyframeCacheMaxEntries;
    int keyframeCacheMaxBytes;
    double localizationSubmapUpdateDistance;
    bool saveLog;
    string saveMapDirectory;
    string saveKeyframeMapDirectory;
    string loadKeyframeMapDirectory;

    bool localizationMode;

    // Lidar Sensor Configuration
    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;   
    
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    int matchingThreads;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    // Added by ACE 2024.4.9
    float surroundingKeyframeSearchRadiusforSharedControl;
    float surroundingLocalMapCropRadiusforSharedControl;
    
    // Loop closure
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    // temporary mapping
    float starttemporaryMappingDistThre; // key poses sparsified by 2.0 m
    float inlierThreshold; // if no serious map out-of-date, it suffice 99% of the time
    float startTemporaryMappingInlierRatioThre; 
    float exitTemporaryMappingInlierRatioThre; 
    int slidingWindowSize;

    float startRelocInlierRatioThre;
    float relocSearchRadius;

    int slidingWindowSizeFusion;
    
    double scDistThres, scMaximumRadius; // added by ACE 2024.4.19
    bool reloc_enabled;

    ParamServer()
    {
        nh.param<double>("roll/globalMatchingRate", globalMatchingRate, 10.0);

        nh.param<bool>("roll/bodyCentered", bodyCentered,false);
        nh.param<bool>("roll/edgeEnabled", edgeEnabled,true);
        nh.param<bool>("roll/gtAvailable", gtAvailable,false);
        nh.param<bool>("roll/overlapFilterEnabled", overlapFilterEnabled,false);

        nh.param<bool>("roll/bashRun", bashRun,false);

        nh.param<std::string>("roll/map_dir", map_dir, "~/Downloads");
        nh.param<std::string>("roll/log_dir", log_dir, "~/Downloads");

        nh.param<bool>("roll/debugMode", debugMode,false);
        nh.param<bool>("roll/useGPS", useGPS,false);

        nh.param<double>("roll/lati0", lati0, 0.0);
        nh.param<double>("roll/longi0", longi0, 0.0);
        nh.param<double>("roll/alti0", alti0, 0.0);

        nh.param<float>("roll/starttemporaryMappingDistThre", starttemporaryMappingDistThre, 20.0);
        nh.param<float>("roll/inlierThreshold", inlierThreshold, 0.1);
        nh.param<float>("roll/startTemporaryMappingInlierRatioThre", startTemporaryMappingInlierRatioThre, 0.4);
        nh.param<float>("roll/exitTemporaryMappingInlierRatioThre", exitTemporaryMappingInlierRatioThre, 0.4);
        nh.param<int>("roll/slidingWindowSize", slidingWindowSize, 30);

        nh.param<float>("roll/startRelocInlierRatioThre", startRelocInlierRatioThre, 0.4);
        nh.param<float>("roll/relocSearchRadius", relocSearchRadius, 10.0);

        nh.param<int>("roll/slidingWindowSizeFusion", slidingWindowSizeFusion,100);

        nh.param<std::string>("/robot_id", robot_id, "roboat");
        nh.param<int>("roll/optIteration", optIteration,30);
        nh.param<float>("roll/degeneracyEigenThreshold", degeneracyEigenThreshold,100.0f);
        if (!std::isfinite(degeneracyEigenThreshold) || degeneracyEigenThreshold <= 0)
            throw std::runtime_error("roll/degeneracyEigenThreshold must be finite and positive");
        nh.param<std::string>("roll/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("roll/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("roll/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("roll/gpsTopic", gpsTopic, "fix");
        nh.param<std::string>("roll/gtTopic", gtTopic, "ground_truth");
        nh.param<std::string>("roll/odometryTopic", odometryTopic, "OdometryLidar");
        

        nh.param<std::string>("roll/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("roll/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("roll/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("roll/mapFrame", mapFrame, "map");

        
        nh.param<bool>("roll/useGpsElevation", useGpsElevation, false);
        nh.param<float>("roll/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("roll/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>("roll/saveLog", saveLog, true);
        nh.param<bool>("roll/savePCD", savePCD, false);
        nh.param<bool>("roll/savePose", savePose, false);
        nh.param<bool>("roll/saveKeyframeMap", saveKeyframeMap, false);
        
        nh.param<bool>("roll/localizationMode", localizationMode, false);
        nh.param<bool>("roll/mapUpdateEnabled", mapUpdateEnabled, false);
        nh.param<bool>("roll/temporaryMappingEnabled", temporaryMappingEnabled, false);
        nh.param<bool>("roll/latestScanOnly", latestScanOnly, false);
        nh.param<bool>("roll/enableSecondaryFusion", enableSecondaryFusion, false);
        nh.param<int>("roll/visualizationPathLimit", visualizationPathLimit, 2000);
        nh.param<bool>("roll/useKeyframeCache", useKeyframeCache, true);
        nh.param<bool>("roll/gravityAlignedMap", gravityAlignedMap, true);
        nh.param<int>("roll/keyframeCacheMaxEntries", keyframeCacheMaxEntries, 256);
        nh.param<int>("roll/keyframeCacheMaxBytes", keyframeCacheMaxBytes, 128*1024*1024);
        nh.param<double>("roll/localizationSubmapUpdateDistance", localizationSubmapUpdateDistance, 0.0);

        
        nh.param<std::string>("roll/saveMapDirectory", saveMapDirectory, "~/Downloads");
        nh.param<std::string>("roll/loadKeyframeMapDirectory", loadKeyframeMapDirectory, "~/Downloads");
        nh.param<std::string>("roll/saveKeyframeMapDirectory", saveKeyframeMapDirectory, "~/Downloads");        


        std::string sensorStr;
        nh.param<std::string>("roll/sensor", sensorStr, "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid sensor type (must be either 'velodyne' or 'ouster'): " << sensorStr);
            ros::shutdown();
        }
        nh.param<int>("roll/gnss_init_num", gnss_init_num, 10);

        nh.param<int>("roll/N_SCAN", N_SCAN, 16);
        nh.param<int>("roll/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>("roll/downsampleRate", downsampleRate, 1);
        nh.param<float>("roll/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>("roll/lidarMaxRange", lidarMaxRange, 1000.0);

        nh.param<vector<float>>("roll/odom_variance", odom_variance, vector<float>(6, 0.01));
        nh.param<vector<float>>("roll/loop_variance", loop_variance, vector<float>(6, 0.5));
        nh.param<vector<float>>("roll/gps_variance", gps_variance, vector<float>(6, 0.01));
        nh.param<vector<float>>("roll/initialGuess", initialGuess, vector<float>(6, 0));
        nh.param<vector<float>>("roll/array_imu_to_body", array_imu_to_body, vector<float>(6, 0));
        nh.param<vector<float>>("roll/array_lidar_to_body", array_lidar_to_body, vector<float>(6, 0));
        nh.param<vector<float>>("roll/array_gps_to_body", array_gps_to_body, vector<float>(6, 0));

        nh.param<float>("roll/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("roll/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("roll/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("roll/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>("roll/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("roll/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("roll/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>("roll/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("roll/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("roll/numberOfCores", numberOfCores, 2);
        nh.param<int>("roll/matchingThreads", matchingThreads, 2);

        nh.param<float>("roll/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>("roll/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("roll/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("roll/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);
        
        nh.param<float>("roll/surroundingKeyframeSearchRadiusforSharedControl", surroundingKeyframeSearchRadiusforSharedControl, 10.0);
        nh.param<float>("roll/surroundingLocalMapCropRadiusforSharedControl", surroundingLocalMapCropRadiusforSharedControl, 5.0);

        nh.param<bool>("roll/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("roll/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("roll/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("roll/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("roll/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("roll/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("roll/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>("roll/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("roll/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("roll/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        // added by ACE 2024.4.19
        nh.param<double>("roll/sc_dist_thres", scDistThres, 0.2);
        nh.param<double>("roll/sc_max_radius", scMaximumRadius, 40.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor
        nh.param<bool>("roll/reloc_enabled", reloc_enabled, false);

        auto checkSix = [](const vector<float>& v, const string& name, bool positive) {
            if (v.size() != 6 || any_of(v.begin(), v.end(), [positive](float x) {
                    return !std::isfinite(x) || (positive && x <= 0);
                })) throw std::runtime_error("Invalid six-element parameter: " + name);
        };
        checkSix(odom_variance, "roll/odom_variance", true);
        checkSix(loop_variance, "roll/loop_variance", true);
        checkSix(initialGuess, "roll/initialGuess", false);
        checkSix(array_imu_to_body, "roll/array_imu_to_body", false);
        checkSix(array_lidar_to_body, "roll/array_lidar_to_body", false);
        checkSix(array_gps_to_body, "roll/array_gps_to_body", false);
        if (N_SCAN <= 0 || Horizon_SCAN <= 0 || downsampleRate <= 0 ||
            numberOfCores <= 0 || matchingThreads <= 0 || mappingCornerLeafSize <= 0 || mappingSurfLeafSize <= 0 ||
            keyframeCacheMaxEntries < 1 || keyframeCacheMaxBytes < 1 ||
            !std::isfinite(localizationSubmapUpdateDistance) || localizationSubmapUpdateDistance < 0 ||
            localizationSubmapUpdateDistance >= surroundingKeyframeSearchRadius)
            throw std::runtime_error("Invalid scan, thread or voxel configuration");
        
        usleep(100);
    }
};


sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0) // this is important since publishing actually takes quite some time
        thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}


void printTrans(std::string message, vector<double> transformIn){
    if (transformIn.size() != 6) return;
    cout<<message<<transformIn[0]<<" "<<transformIn[1]<<" "
        <<transformIn[2]<<" "<<transformIn[3]<<" "<<transformIn[4]<<" "<<transformIn[5];
    return;
}
// returning c array is not easy to pull off, just forget it
void Affine3f2Trans(Eigen::Affine3f t,float transformOut[6])
{
    pcl::getTranslationAndEulerAngles(t,transformOut[3], transformOut[4], transformOut[5], transformOut[0], transformOut[1], transformOut[2]);
}

Eigen::Affine3f trans2Affine3f(float transformIn[6])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

Eigen::Affine3f trans2Affine3f(const vector<float> &transformIn)
{
    if(transformIn.size() != 6) return Eigen::Affine3f::Identity();
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

void printTrans(std::string message, const float transformIn[]){
    ROS_INFO_STREAM(message<<transformIn[0]<<" "<<transformIn[1]<<" "
        <<transformIn[2]<<" "<<transformIn[3]<<" "<<transformIn[4]<<" "<<transformIn[5]);
    return;
}

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}

#endif
