#pragma once
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <map>
#include <deque>
#include <queue>
#include <mutex>
#include <thread>
#include <filesystem>
#include <condition_variable>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unistd.h>
using namespace std;
typedef pcl::PointXYZI PointType;
enum class SensorType {VELODYNE,OUSTER,LIVOX};
class YamlParameters {
    YAML::Node root;
public:
    explicit YamlParameters(const YAML::Node& node):root(node){}
    template<class T,class U>void param(const string& key,T& result,const U& fallback)const {
        auto node=YAML::Clone(root);size_t begin=key[0]=='/'?1:0;
        while(begin<key.size()){auto end=key.find('/',begin);auto child=node[key.substr(begin,end-begin)];
            node.reset(child);if(!node)break;if(end==string::npos)break;begin=end+1;}
        result=node?node.as<T>():T(fallback);
    }
};
class ParamServer {
public:
#include "ParameterFields.inc"
    YamlParameters nh;
    explicit ParamServer(const YAML::Node& config):nh(config){
#include "ParameterLoad.inc"
    }
};
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

inline void printTrans(std::string, const float[]) {}

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}
