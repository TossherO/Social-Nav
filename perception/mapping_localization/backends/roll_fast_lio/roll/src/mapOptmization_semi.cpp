#include "utility.h"
#include "roll/cloud_info.h"
#include "roll/save_map.h"
#include "getCpuOccupancy.h"
#include "gnssLiDARinitializer.h"

#include"LOAMmapping.h"
#include "globalOpt.h"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <pcl/filters/passthrough.h>
#include <std_msgs/Int8.h>

using namespace gtsam;

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;
typedef geometry_msgs::PoseWithCovarianceStampedConstPtr rvizPoseType;


class mapOptimization : public ParamServer
{

public:
    // for trajectory alignment
    std::shared_ptr<GlobalOptimization> global_estimator;
    std::shared_ptr<gnssLiDARinitializer> gnss_lidar_initializer;    

    long long cnt = 0;

    int current_pid ;//or you can set a outside program pid
    float cpu_usage_ratio;
    float memory_usage;

    string logTime = "";

    ofstream pose_log_file;
    float fitnessScore = 0;
    ofstream gps_file;

    Eigen::Affine3f lastOdometryPose;
    nav_msgs::Path global_path_gtsam;

    // indoor outdoor keyframe detection
    vector<int> isIndoorKeyframe;
    vector<int> isIndoorKeyframeTMM;

    Eigen::Affine3f affine_imu_to_lidar;
    Eigen::Affine3f affine_lidar_to_gps;
    Eigen::Affine3f affine_imu_to_body;
    Eigen::Affine3f affine_lidar_to_body;
    Eigen::Affine3f affine_gps_to_body;
    //gps
    double rns;
    double rew;
    bool relocSuccess = false;
    bool tryReloc = false; 
    bool mapLoaded = false;

    // for temporary mapping mode
    double rosTimeStart = -1;
    bool temporaryMappingMode = false;
    float transformBeforeMapped[6];
    bool goodToMergeMap = false;
    int startTemporaryMappingIndex = -1;
    float mergeNoise;
    bool frameTobeAbandoned = false;
    
    int TMMcount = 0;
    
    float TMMx = 0,TMMy = 0;
    float maxErrorX = 0, maxErrorY = 0, maxError = -1.0;


    // // use intensity channel of K pointclouds to record the error counts: K=5 for now
    // int K = 5;
    // vector<pcl::PointCloud<PointType>> errorCounts; // not the best way to do this 

    // data analysis
    int iterCount = 0;
    float mappingTime = 0;
    int edgePointCorrNum = 0;
    int surfPointCorrNum = 0;
    float maxEdgeIntensity = -1;
    float maxSurfIntensity = -1;
    float maxIntensity = -1;

    Eigen::Affine3f correctedPose;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    bool doneSavingMap = false;

    Eigen::Affine3f affine_imu_to_odom; // convert points in lidar frame to odom frame
    Eigen::Affine3f affine_imu_to_map;
    Eigen::Affine3f affine_odom_to_map;

    vector<vector<double>> mappingLogs;
    vector<float> noiseVec;
    pcl::PointCloud<PointType>::Ptr submap;
    vector<rvizPoseType> poseEstVec;
    Eigen::Affine3f relocCorrection;

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLidarCloudSurround;
    ros::Publisher pubLidarOdometryGlobal;
    ros::Publisher pubLidarOdometryGlobalFusion;
    ros::Publisher vinsFusion;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;
    ros::Publisher pubPathFusion;
    ros::Publisher pubPathFusionVINS;
    ros::Publisher pubLidarOdometryGlobalFusionBaselink;
    ros::Publisher pubPathGPS;
    ros::Publisher gpsTrajPub;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubMergedMap;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubLoopConstraintEdge;
    ros::Publisher pubKeyPosesTmp;

    ros::Publisher pubRecentKeyFramesforSharedControl; // Added by ACE 2024.4.9

    ros::Subscriber subrslidar;
    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subGT;
    ros::Subscriber subLoop;
    ros::Subscriber subLidarOdometry;
    ros::Subscriber initialpose_sub;
    ros::ServiceServer srvSaveMap;

    ros::Subscriber subGlobalState;
    int global_state_;

    std::deque<nav_msgs::Odometry> gpsQueue;
    
    roll::cloud_info cloudInfo;
    queue<roll::cloud_infoConstPtr> cloudInfoBuffer;
    queue<nav_msgs::Odometry::ConstPtr> lidarOdometryBuffer;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    vector<pcl::PointCloud<PointType>::Ptr> temporaryCornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> temporarySurfCloudKeyFrames;


    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr temporaryCloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr temporaryCloudKeyPoses6D;


    pcl::PointCloud<PointType>::Ptr lidarCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr lidarCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr lidarCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr lidarCloudSurfLastDS; // downsampled surf featuer set from odoOptimization



    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> lidarCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr lidarCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr lidarCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr lidarCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr lidarCloudSurfFromMapDS;



    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    pcl::VoxelGrid<PointType> downSizeFilterSavingKeyframes; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLidarInfoStamp;
    double cloudInfoTime;

    float transformTobeMapped[6];
    
    std::mutex mtx;
    std::mutex mtxInit;
    std::mutex mtxLoopInfo;
    std::mutex pose_estimator_mutex;
    std::mutex mtxGPSqueue;


    int lidarCloudCornerFromMapDSNum = 0;
    int lidarCloudSurfFromMapDSNum = 0;
    int lidarCloudCornerLastDSNum = 0;
    int lidarCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    
    multimap<int,int>    loopIndexContainer;
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    
    deque<std_msgs::Float64MultiArray> loopInfoVec;
    vector<nav_msgs::Odometry> globalOdometry;
    
    nav_msgs::Path globalPath;
    nav_msgs::Path globalPathFusion;
    nav_msgs::Path globalPathFusionVINS;
    nav_msgs::Path globalPathGPS;


    bool poseGuessFromRvizAvailable = false;
    float rvizGuess[6];

    
    pcl::PointCloud<PointType>::Ptr lidarCloudRaw; 
    pcl::PointCloud<PointType>::Ptr rslidar_raw; 

// one keyframe per 1m, considering "0.5m & 0.1 deg"drift per 100m, for every keyframe
    // translation: 0.5/100=5e-3 m
    // rotation: 0.1/57.3/100 = 2e-5
    // Variances, not Covariances!!!
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr corrNoise; // used for loop closusre and prior noise in merging

    mapOptimization()
    {
        
        // from the BOOK, it is standard deviation!!!
        odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << odom_variance[0],odom_variance[1],odom_variance[2],
                    odom_variance[3],odom_variance[4],odom_variance[5]).finished());
        corrNoise = noiseModel::Diagonal::Variances((Vector(6) << loop_variance[0],loop_variance[1],loop_variance[2],
                    loop_variance[3],loop_variance[4],loop_variance[5]).finished()); 

        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);   
        long long time_sec = mktime(t_tm);
        logTime = to_string(time_sec);
        

        // gps parameter calculation
        double earthEqu = 6378135;
        double earthPolar = 6356750;
        double tmp = sqrt(earthEqu*earthEqu*cos(deg2rad(lati0))*cos(deg2rad(lati0)) +earthPolar*earthPolar*sin(deg2rad(lati0))*sin(deg2rad(lati0)));
        rns = earthEqu*earthEqu*earthPolar*earthPolar/tmp/tmp/tmp;
        rew = earthEqu*earthEqu/tmp;

        gpsTrajPub      = nh.advertise<nav_msgs::Odometry> ("/semi/roll/gps_odom", 1);

        pubPath                     = nh.advertise<nav_msgs::Path>("/semi/roll/mapping/path", 1);
        pubPathFusion               = nh.advertise<nav_msgs::Path>("/semi/roll/mapping/path_fusion", 1);
        pubPathFusionVINS               = nh.advertise<nav_msgs::Path>("/semi/roll/mapping/path_fusion_vins", 1);
        pubLidarOdometryGlobalFusionBaselink      = nh.advertise<nav_msgs::Odometry> ("/semi/roll/mapping/odometry_fusion_baselink", 1);
        pubPathGPS                     = nh.advertise<nav_msgs::Path>("/semi/roll/mapping/gps", 1);

        // initialpose_sub = nh.subscribe("/initialpose", 1, &mapOptimization::initialpose_callback, this);

        pubKeyPosesTmp                 = nh.advertise<sensor_msgs::PointCloud2>("/semi/roll/mapping/tmp_key_poses", 1);
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("/semi/roll/mapping/key_poses", 1);
        pubLidarCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("/semi/roll/mapping/map_global", 1);
        pubLidarOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("/semi/roll/mapping/odometry", 1);
        pubLidarOdometryGlobalFusion      = nh.advertise<nav_msgs::Odometry> ("/semi/roll/mapping/odometry_fusion", 1);
        vinsFusion      = nh.advertise<nav_msgs::Odometry> ("/semi/roll/mapping/vins_fusion", 1);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("/semi/roll/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("/semi/roll/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/semi/roll/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("/semi/roll/mapping/map_local", 1);

        pubRecentKeyFramesforSharedControl = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/localmap_for_sharedcontrol", 1);

        pubMergedMap = nh.advertise<sensor_msgs::PointCloud2>("/semi/roll/mapping/merged_map", 1);

        // pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/cloud_registered_semi", 1);
        subrslidar = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 10, &mapOptimization::lidarhandler, this);
        subCloud = nh.subscribe<roll::cloud_info>("/roll/feature/cloud_info", 10, &mapOptimization::lidarCloudInfoHandler, this);
        // subGPS   = nh.subscribe<sensor_msgs::NavSatFix> (gpsTopic, 200, &mapOptimization::gpsHandler, this);
        // subGT   = nh.subscribe<nav_msgs::Odometry> (gtTopic, 200, &mapOptimization::gtHandler, this); 
        subLidarOdometry = nh.subscribe<nav_msgs::Odometry> (odometryTopic, 10, &mapOptimization::lidarOdometryHandler,this);

        subGlobalState = nh.subscribe<std_msgs::Int8> ("robot_state", 1, &mapOptimization::GlobalStatusCallback,this);
        global_state_ = 0;

        // srvSaveMap  = nh.advertiseService("/roll/save_map", &mapOptimization::saveMapService, this);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization


        allocateMemory();

        // string log_file_name;
        // if(bashRun) log_file_name = log_dir + "/logs/log"+ logTime + ".txt";
        // else log_file_name = saveKeyframeMapDirectory + "/../log.txt";
        // pose_log_file.open(log_file_name);
        // pose_log_file.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
        // pose_log_file.precision(6); // 固定小数位6
        // if(!pose_log_file.is_open())
        // {
        //     std::cout<<"Cannot open"<<log_file_name<<endl;
        // }
        // else std::cout<<"Logging to "<<log_file_name<<endl;
        // gps_file.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
        // gps_file.precision(6); // 固定小数位6

        // string gps_file_name;
        // if(bashRun) gps_file_name = log_dir +"/gps/gps"+ logTime + ".txt";
        // else gps_file_name = saveKeyframeMapDirectory +"/../gps.txt";

        // gps_file.open(gps_file_name,ios::out);
        // if(!gps_file.is_open())
        // {
        //     std::cout<<"Cannot open"<<gps_file_name<<endl;
        // }
        //         else std::cout<<"gps log saved to "<<gps_file_name<<endl;

        // if (localizationMode) // load the keyframes and construct a map
        // { // even ctrl+C won't terminate loading process
        //     std::lock_guard<std::mutex> lock(mtxInit);
        //     // load keyframe map
        //     if (!mapLoaded){
        //         ROS_INFO("************************loading keyframe map************************");
        //         string filePath;
        //         if(bashRun) saveKeyframeMapDirectory = map_dir+"/keyframes";

        //         ifstream fin(saveKeyframeMapDirectory+"/poses.txt");
        //         if (!fin.is_open()) {
        //             std::cout<<saveKeyframeMapDirectory<<" is not valid!"<<endl;
        //         }
        //         bool first_pose = true;
        //         int is_indoor = 0;

        //         // adapt for the following 3 kinds of map:
        //         //1. xyzrpy index
        //         //2. xyzrpy index is_indoor
        //         //3. time xyzrpy intensity index
        //         vector<double> data_tmp;
        //         while (true){
        //             PointTypePose point;
        //             PointType point3;
        //             if(first_pose == true)
        //             {
        //                 string tmp_str;
        //                 getline(fin,tmp_str);
        //                 istringstream iss(tmp_str);
        //                 string tmp_str2;
        //                 while(iss>>tmp_str2)
        //                 {
        //                     data_tmp.push_back(stod(tmp_str2));
        //                 }
        //                 if(data_tmp.size() == 7 || data_tmp.size() == 8 )
        //                 {
        //                     point.x=data_tmp[0];
        //                     point.y=data_tmp[1];
        //                     point.z=data_tmp[2];
        //                     point.roll=data_tmp[3];
        //                     point.pitch=data_tmp[4];
        //                     point.yaw=data_tmp[5];
        //                     point.intensity=data_tmp[6];
        //                     if(data_tmp.size() == 8) is_indoor = data_tmp[7];
        //                     first_pose = false;
        //                 }
        //                 else if(data_tmp.size() == 9)
        //                 {
        //                     point.time = data_tmp[0];
        //                     point.x=data_tmp[1];
        //                     point.y=data_tmp[2];
        //                     point.z=data_tmp[3];
        //                     point.roll=data_tmp[4];
        //                     point.pitch=data_tmp[5];
        //                     point.yaw=data_tmp[6];
        //                     point.intensity=data_tmp[7];
        //                     is_indoor = data_tmp[8];
        //                     first_pose = false;
        //                 }
        //                 else
        //                 {
        //                     ROS_ERROR("wrong format for poses.txt!");
        //                     break;
        //                 }
                        
        //             }
        //             else
        //             {
        //                 if(data_tmp.size() == 7)
        //                     fin>>point.x>>point.y>>point.z>>point.roll>>point.pitch>>point.yaw>>point.intensity;
        //                 else if (data_tmp.size() == 8)
        //                 {
        //                     fin>>point.x>>point.y>>point.z>>point.roll>>point.pitch>>point.yaw>>point.intensity>>is_indoor;
        //                 }
        //                 else
        //                 {
        //                     fin>>point.time>>point.x>>point.y>>point.z>>point.roll>>point.pitch>>point.yaw>>point.intensity>>is_indoor;
        //                 }
        //             }
        //             point3.x=point.x;
        //             point3.y=point.y;
        //             point3.z=point.z;
        //             point3.intensity=point.intensity;                    
        //             if(fin.peek()==EOF)
        //             { 
        //                 break;
        //             }
        //             else{
        //                 cloudKeyPoses6D->push_back(point);
        //                 cloudKeyPoses3D->push_back(point3);
        //                 isIndoorKeyframe.push_back(is_indoor);
        //             }
        //         }
        //         int keyframeN = (int)cloudKeyPoses6D->size();
        //         ROS_INFO("There are in total %d keyframes",keyframeN);
        //         for (int i=0;i<keyframeN;i++){
        //             pcl::PointCloud<PointType>::Ptr cornerKeyFrame(new pcl::PointCloud<PointType>());
        //             pcl::PointCloud<PointType>::Ptr surfKeyFrame(new pcl::PointCloud<PointType>());
        //             string cornerFileName = saveKeyframeMapDirectory + "/corner"+ to_string(i) + ".pcd";
        //             string surfFileName = saveKeyframeMapDirectory + "/surf"+ to_string(i) + ".pcd";

        //             if (pcl::io::loadPCDFile<PointType> (cornerFileName, *cornerKeyFrame) == -1) 
        //                std::cout<< "Couldn't read file"+ cornerFileName <<endl;
        //             if (pcl::io::loadPCDFile<PointType> (surfFileName, *surfKeyFrame) == -1) 
        //                std::cout<< "Couldn't read file"+ surfFileName <<endl;
        //             cornerCloudKeyFrames.push_back(cornerKeyFrame);
        //             surfCloudKeyFrames.push_back(surfKeyFrame);
        //             if (i%100 == 0)
        //                 std::cout << "\r" << std::flush << "Loading feature cloud " << i << " of " << keyframeN-1 << " ...\n";
        //         }
        //         ROS_INFO("************************Keyframe map loaded************************");
        //         mapLoaded=true;
        //     }
        // }
        
    }

    void allocateMemory()
    {        
        
        
        resetISAM();
        affine_imu_to_body = trans2Affine3f(array_imu_to_body);
        affine_lidar_to_body =  trans2Affine3f(array_lidar_to_body);
        affine_gps_to_body =  trans2Affine3f(array_gps_to_body);

        affine_imu_to_lidar = affine_lidar_to_body.inverse()*affine_imu_to_body;
        
        affine_lidar_to_gps = affine_gps_to_body.inverse()*affine_lidar_to_body;
        std::cout<<"affine_lidar_to_gps: "<<affine_lidar_to_gps.matrix()<<endl;

        global_estimator = std::make_shared<GlobalOptimization>(slidingWindowSizeFusion);
        gnss_lidar_initializer = std::make_shared<gnssLiDARinitializer>(gnss_init_num);

        affine_imu_to_map = Eigen::Affine3f::Identity();
        affine_imu_to_odom = Eigen::Affine3f::Identity();
        affine_odom_to_map = Eigen::Affine3f::Identity();
        correctedPose = Eigen::Affine3f::Identity();

        submap.reset(new pcl::PointCloud<PointType>()); // why dot when it is pointer type

        lidarCloudRaw.reset(new pcl::PointCloud<PointType>()); 

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        temporaryCloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        temporaryCloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        lidarCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        lidarCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        lidarCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        lidarCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization


        lidarCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        lidarCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        lidarCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        lidarCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        for (int i = 0; i < 6; ++i){
            transformBeforeMapped[i] = 0;
            transformTobeMapped[i] = 0;
        }
        
        // cv::Mat matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    } 

    void odometryMsgToAffine3f(nav_msgs::Odometry msgIn,Eigen::Affine3f &trans)
    {
        tf::Quaternion tfQ(msgIn.pose.pose.orientation.x,msgIn.pose.pose.orientation.y,msgIn.pose.pose.orientation.z,msgIn.pose.pose.orientation.w);
        double roll,pitch,yaw;
        tf::Matrix3x3(tfQ).getRPY(roll,pitch,yaw);
        // think about why not update affine_imu_to_odom and affine_imu_to_map here!!!
        trans = pcl::getTransformation(msgIn.pose.pose.position.x,
        msgIn.pose.pose.position.y,msgIn.pose.pose.position.z, float(roll),float(pitch),float(yaw));

        
    }
     void odometryMsgToTrans(const nav_msgs::Odometry::ConstPtr& msgIn,float trans[6])
    {
        tf::Quaternion tfQ(msgIn->pose.pose.orientation.x,msgIn->pose.pose.orientation.y,msgIn->pose.pose.orientation.z,msgIn->pose.pose.orientation.w);
        double roll,pitch,yaw;
        tf::Matrix3x3(tfQ).getRPY(roll,pitch,yaw);
        trans[0] = roll;
        trans[1] = pitch;
        trans[2] = yaw;
        trans[3] = msgIn->pose.pose.position.x;
        trans[4] = msgIn->pose.pose.position.y;
        trans[5] = msgIn->pose.pose.position.z;
    }

    void GlobalStatusCallback(const std_msgs::Int8ConstPtr& global_state)
    {
      mtx.lock();
      if (global_state->data == 2) {
        std::cout << "****************************** Activate Semi Auto Mode ******************************" << std::endl;

        global_state_ = 2;
      } else {
        std::cout << "****************************** Deactivate Semi Auto Mode ******************************" << std::endl;
        
        global_state_ = 0;

        queue<roll::cloud_infoConstPtr> empty_queue1;
        swap(cloudInfoBuffer, empty_queue1);
        queue<nav_msgs::Odometry::ConstPtr> empty_queue2;
        swap(lidarOdometryBuffer, empty_queue2);
        // cornerCloudKeyFrames.clear();
        // surfCloudKeyFrames.clear();
        // allocateMemory();
      }
      mtx.unlock();
    }

    void lidarCloudInfoHandler(const roll::cloud_infoConstPtr& msgIn)
    {
      if (global_state_ == 2) {
        mtx.lock();
        cloudInfoBuffer.push(msgIn);
        mtx.unlock();
      }
    }
    void lidarhandler(const sensor_msgs::PointCloud2ConstPtr &lidarCloudMsg )
    {
        rslidar_raw.reset(new pcl::PointCloud<PointType>()); 
        pcl::fromROSMsg(*lidarCloudMsg,*rslidar_raw);
    }

    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& msgIn)
    {
      if (global_state_ == 2) {
        mtx.lock();
        lidarOdometryBuffer.push(msgIn);
        mtx.unlock();
        
        // Publish TF
        Eigen::Affine3f affine_lidar_to_imu0,affine_lidar_to_lidar0;
        odometryMsgToAffine3f(*msgIn, affine_lidar_to_imu0);

        // fastlio "camera_init" is the 1st imu frame now when need_map_erected is set false
        // for nclt, only need incremental deltaT, affine_imu_to_lidar can be whatever
        affine_lidar_to_lidar0 = affine_imu_to_lidar*affine_lidar_to_imu0;
        //odom to map!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Eigen::Affine3f affine_lidar_to_map_tmp = affine_odom_to_map*affine_lidar_to_lidar0;
        float array_imu_to_map[6];
        Affine3f2Trans(affine_lidar_to_map_tmp, array_imu_to_map);
        tf::Quaternion q_imu_to_map = tf::createQuaternionFromRPY(array_imu_to_map[0],array_imu_to_map[1],array_imu_to_map[2]);

        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(array_imu_to_map[0], array_imu_to_map[1], array_imu_to_map[2]),
                                                      tf::Vector3(array_imu_to_map[3], array_imu_to_map[4], array_imu_to_map[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, msgIn->header.stamp, mapFrame, lidarFrame);
        // br.sendTransform(trans_odom_to_lidar);

        tf::Transform t_odom_to_lidar_foot = tf::Transform(tf::createQuaternionFromRPY(0,0, array_imu_to_map[2]),tf::Vector3(array_imu_to_map[3],
            array_imu_to_map[4], 0.2));
        tf::StampedTransform trans_odom_to_lidar_foot = tf::StampedTransform(t_odom_to_lidar_foot, msgIn->header.stamp, mapFrame, "lidar_footprint");
        // br.sendTransform(trans_odom_to_lidar_foot);
        
        // in loc mode, publish lidar_to_map tf so pointcloud can be visualized in rviz
        if (relocSuccess == false) return;

        global_estimator->inputOdom(msgIn->header.stamp.toSec(),affine_lidar_to_lidar0.matrix().cast<double>());
        // testing fusion
        Eigen::Vector3d odomP;
        Eigen::Quaterniond odomQ;
        global_estimator->getGlobalOdom(odomP, odomQ);

        nav_msgs::Odometry odomFusion;
        odomFusion.header.frame_id = mapFrame;
        odomFusion.header.stamp = msgIn->header.stamp;
        odomFusion.pose.pose.orientation.x = odomQ.x();
        odomFusion.pose.pose.orientation.y = odomQ.y();
        odomFusion.pose.pose.orientation.z = odomQ.z();
        odomFusion.pose.pose.orientation.w = odomQ.w();
        odomFusion.pose.pose.position.x = odomP[0];
        odomFusion.pose.pose.position.y = odomP[1];
        odomFusion.pose.pose.position.z = odomP[2];
        vinsFusion.publish(odomFusion);

        geometry_msgs::PoseStamped pose_stampedF;
        pose_stampedF.pose = odomFusion.pose.pose;
        pose_stampedF.header.frame_id = mapFrame;
        pose_stampedF.header.stamp = msgIn->header.stamp;
        globalPathFusionVINS.poses.push_back(pose_stampedF);
        globalPathFusionVINS.header.stamp = msgIn->header.stamp;
        globalPathFusionVINS.header.frame_id = mapFrame;

        if(pubPathFusionVINS.getNumSubscribers() != 0)   
            pubPathFusionVINS.publish(globalPathFusionVINS); // before loop closure

        tf::Transform tf_lidar_to_baselink = tf::Transform(tf::createQuaternionFromRPY(0,0,0),tf::Vector3(-0.67, -0.325,0)); 
        tf::Transform t_odom_to_baselink = t_odom_to_lidar*tf_lidar_to_baselink;

        double r,p,y;
        tf::Quaternion q(t_odom_to_baselink.getRotation().x(),t_odom_to_baselink.getRotation().y(),t_odom_to_baselink.getRotation().z()
            ,t_odom_to_baselink.getRotation().w());
        tf::Matrix3x3(q).getRPY(r,p,y);
        // std::cout<<"odom to baselink rpy XYZ: "<<r<<" "<<p<<" "<<y<<" "<<t_odom_to_baselink.getOrigin()[0]<<" "<<t_odom_to_baselink.getOrigin()[1]
            // <<" "<<t_odom_to_baselink.getOrigin()[2]<<endl;

        nav_msgs::Odometry odomAftMappedBaselink;
        odomAftMappedBaselink.header.frame_id = mapFrame;
        odomAftMappedBaselink.header.stamp = msgIn->header.stamp;
        odomAftMappedBaselink.pose.pose.orientation.x = t_odom_to_baselink.getRotation().x();
        odomAftMappedBaselink.pose.pose.orientation.y =  t_odom_to_baselink.getRotation().y();
        odomAftMappedBaselink.pose.pose.orientation.z =  t_odom_to_baselink.getRotation().z();
        odomAftMappedBaselink.pose.pose.orientation.w =  t_odom_to_baselink.getRotation().w();
        odomAftMappedBaselink.pose.pose.position.x =  t_odom_to_baselink.getOrigin()[0];
        odomAftMappedBaselink.pose.pose.position.y = t_odom_to_baselink.getOrigin()[1];
        odomAftMappedBaselink.pose.pose.position.z = t_odom_to_baselink.getOrigin()[2];
        pubLidarOdometryGlobalFusionBaselink.publish(odomAftMappedBaselink);
        

        float odomTmp[6];
        Affine3f2Trans(affine_lidar_to_lidar0, odomTmp);
        // std::cout<<"odom message: "<<odomTmp[3]<<" "<< odomTmp[4]<<endl;
        
        nav_msgs::Odometry odomAftMapped;
        odomAftMapped.header.frame_id = mapFrame;
        odomAftMapped.header.stamp = msgIn->header.stamp;
        odomAftMapped.pose.pose.orientation.x = q_imu_to_map.x();
        odomAftMapped.pose.pose.orientation.y = q_imu_to_map.y();
        odomAftMapped.pose.pose.orientation.z = q_imu_to_map.z();
        odomAftMapped.pose.pose.orientation.w = q_imu_to_map.w();
        odomAftMapped.pose.pose.position.x = array_imu_to_map[3];
        odomAftMapped.pose.pose.position.y = array_imu_to_map[4];
        odomAftMapped.pose.pose.position.z = array_imu_to_map[5];
        odomAftMapped.twist = msgIn->twist;
        pubLidarOdometryGlobalFusion.publish(odomAftMapped);

        // // sometimes it cannot find tf when bagtime is not the same with the sensor time
        // tf::TransformListener tf_;
        // tf::StampedTransform stamped_lidar_to_baselink;
        // try
        // {
        //     tf_.lookupTransform(baselinkFrame, lidarFrame, msgIn->header.stamp, stamped_lidar_to_baselink);
        // }
        // catch(tf::TransformException e)
        // {
        //     ROS_ERROR("Failed to compute lidar_to_baselink: (%s)", e.what());
        //     return;
        // }
        // tf::Transform t_odom_to_baselink = t_odom_to_lidar*stamped_lidar_to_baselink;
        // child frame 'lidar_link' expressed in parent_frame 'mapFrame'
        // tf::StampedTransform trans_odom_to_baselink = tf::StampedTransform(t_odom_to_baselink, msgIn->header.stamp, mapFrame, baselinkFrame);
        // br.sendTransform(trans_odom_to_baselink);
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.pose = odomAftMapped.pose.pose;
        pose_stamped.header.frame_id = mapFrame;
        pose_stamped.header.stamp = msgIn->header.stamp;
        globalPathFusion.poses.push_back(pose_stamped);
        globalPathFusion.header.stamp = msgIn->header.stamp;
        globalPathFusion.header.frame_id = mapFrame;
        if(pubPathFusion.getNumSubscribers() != 0) 
            pubPathFusion.publish(globalPathFusion); // before loop closure
      }
    }

    void transformUpdate()
    {
        mtx.lock();
        affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse();
        mtx.unlock();
    }

    void run()
    {
        ros::Rate matchingRate(globalMatchingRate);
        // ROS_INFO("%d",global_state_);
        while(ros::ok()){ // why while(1) is not okay???
            if (global_state_ == 2) { // semi-auto mode
            // ROS_INFO("1");
            while (!cloudInfoBuffer.empty() && !lidarOdometryBuffer.empty())
            {
                mtx.lock();
                while (!lidarOdometryBuffer.empty() && lidarOdometryBuffer.front()->header.stamp.toSec() < cloudInfoBuffer.front()->header.stamp.toSec())
                {
                    lidarOdometryBuffer.pop();
                }
                if (lidarOdometryBuffer.empty()){
                    mtx.unlock();
                    break;
                }
                timeLidarInfoStamp = cloudInfoBuffer.front()->header.stamp;
                cloudInfoTime = cloudInfoBuffer.front()->header.stamp.toSec();
                double lidarOdometryTime = lidarOdometryBuffer.front()->header.stamp.toSec();

                if(debugMode) std::cout<<setiosflags(ios::fixed)<<setprecision(3)<<"cloud time: "<<cloudInfoTime-rosTimeStart<<endl;

                if (rosTimeStart < 0) rosTimeStart = cloudInfoTime;

                if (abs(lidarOdometryTime - cloudInfoTime) > 0.05) // normally >, so pop one cloud_info msg
                {
                    // ROS_WARN("Unsync message!");
                    cloudInfoBuffer.pop();  // pop the old one,otherwise it  will go to dead loop, different from aloam 
                    mtx.unlock();
                    break;
                }

                // extract info and feature cloud
                roll::cloud_infoConstPtr cloudInfoMsg = cloudInfoBuffer.front();
                nav_msgs::Odometry::ConstPtr lidarOdometryMsg =  lidarOdometryBuffer.front();

                lidarCloudRaw.reset(new pcl::PointCloud<PointType>()); 
                cloudInfo = *cloudInfoMsg;
                Eigen::Affine3f tmp;
                odometryMsgToAffine3f(*lidarOdometryMsg,tmp);

                //fastlio use 'camera_init' (i.e. the 1st imu frame)as odom frame, convert to 1st lidar frame
                affine_imu_to_odom = affine_imu_to_lidar*tmp;

                pcl::fromROSMsg(cloudInfoMsg->cloud_corner,  *lidarCloudCornerLast);
                pcl::fromROSMsg(cloudInfoMsg->cloud_surface, *lidarCloudSurfLast);
                pcl::fromROSMsg(cloudInfoMsg->cloud_raw,  *lidarCloudRaw);
                // clear
                lidarOdometryBuffer.pop(); 
                while (!cloudInfoBuffer.empty())
                {
                    cloudInfoBuffer.pop();
                    // ROS_INFO_STREAM("popping old cloud_info messages for real-time performance");
                }
                mtx.unlock();

                TicToc mapping;

                updateInitialGuess(); // actually the same as ALOAM

                if (tryReloc == true || relocSuccess == true)
                {
                    
                    TicToc extract;
                    extractNearby();
                    
                    if(debugMode) std::cout<<"extract: "<<extract.toc()<<endl;
                    TicToc downsample;
                    downsampleCurrentScan();

                    if(debugMode) std::cout<<"downsample: "<<downsample.toc()<<endl;
                    TicToc opt;
                    
                    scan2MapOptimization();
                    
                    float optTime = opt.toc();
                    if(debugMode)  std::cout<<"optimization: "<<optTime<<endl; // > 90% of the total time
                    
                    TicToc optPose;
                    // if (localizationMode)
                    // {
                    //     saveTemporaryKeyframes();
                    //     updatePathRELOC(cloudInfoMsg);  // for visualizing in rviz
                    // }
                    // else
                    {
                        saveKeyFramesAndFactor();
                        correctPoses();
                    }
                    float optPoseTime = optPose.toc();
                    if(debugMode)  std::cout<<"pose opt. takes "<< optPoseTime<<endl;
                

                }
                
                // TicToc publish;
                publishLocalMap();
                publishLocalMapforSharedControl();
                publishOdometry();
                transformUpdate();
                
                frameTobeAbandoned = false;
                if(debugMode) printTrans("after mapping: ",transformTobeMapped);
                // std::cout<<"publish: "<<publish.toc()<<endl;
                // std::cout<<"after mapping z diff: "<<transformTobeMapped[2]-transformBeforeMapped[2]<<endl;


                // ROS_INFO_STREAM("At time "<< cloudInfoTime - rosTimeStart);
                if (goodToMergeMap)
                {
                    if (mapUpdateEnabled)
                        mergeMap();
                    // downsize temporary maps to slidingWindowSize
                    auto iteratorKeyPoses3D = temporaryCloudKeyPoses3D->begin();
                    auto iteratorKeyPoses6D = temporaryCloudKeyPoses6D->begin();
                    auto iteratorKeyFramesC = temporaryCornerCloudKeyFrames.begin();
                    auto iteratorKeyFramesS = temporarySurfCloudKeyFrames.begin();
                    auto iteratorKeyFramesI = isIndoorKeyframeTMM.begin();
                    // usually added cloud would not be big so just leave the sparsification to savingMap
                    // ROS_INFO_STREAM("At time "<< cloudInfoTime - rosTimeStart<< " sec, Merged map has "<<(int)temporaryCloudKeyPoses3D->size()<< " key poses");
                    while ((int)temporaryCloudKeyPoses3D->size() > slidingWindowSize)
                    {
                        // automatically +1
                        temporaryCloudKeyPoses3D->erase(iteratorKeyPoses3D);
                        temporaryCloudKeyPoses6D->erase(iteratorKeyPoses6D);
                        temporaryCornerCloudKeyFrames.erase(iteratorKeyFramesC);
                        temporarySurfCloudKeyFrames.erase(iteratorKeyFramesS);
                        isIndoorKeyframeTMM.erase(iteratorKeyFramesI);
                    }
                    // std::cout<<temporaryCloudKeyPoses3D->size()<<endl;
                    // std::cout<<"reindexing: key poses and key frames are corresponding with respect to the adding sequence"<<endl;
                    for (int i = 0 ; i< (int)temporaryCloudKeyPoses3D->size(); i++)
                    {
                        temporaryCloudKeyPoses3D->points[i].intensity = i;
                        temporaryCloudKeyPoses6D->points[i].intensity = i;
                    }
                    goodToMergeMap = false;
                    temporaryMappingMode = false;
                }

                float mappingTimeCur = mapping.toc();
                mappingTime = (mappingTime*cnt+mappingTimeCur)/(cnt+1);
                cnt++;
                if(debugMode)  std::cout<<"mapping time: "<<mappingTimeCur<<endl;

                if(globalMatchingRate <= 4.0) // otherwise it is just waste of time
                    matchingRate.sleep();
            }


            }
        }

    }

    void mergeMap()
    {
        std::cout<<" DO gtsam optimization here"<<endl;
        TicToc t_merge;
        int priorNode = 0;
        
        // gtsam
        NonlinearFactorGraph gtSAMgraphTM;
        Values initialEstimateTM;        
        ISAM2 *isamTM;
        Values isamCurrentEstimateTM;
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isamTM = new ISAM2(parameters);

        gtsam::Pose3 posePrior = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[priorNode]);
        gtSAMgraphTM.add(PriorFactor<Pose3>(priorNode, posePrior, corrNoise));
        initialEstimateTM.insert(priorNode, posePrior);

        int tempSize = temporaryCloudKeyPoses6D->points.size();

        if (tempSize < 3 ) return;
        for (int i = priorNode; i < tempSize - 2; i++)
        {
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[i]);
            gtsam::Pose3 poseTo   = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[i+1]);
            gtSAMgraphTM.add(BetweenFactor<Pose3>(i,i+1, poseFrom.between(poseTo), odometryNoise));
            initialEstimateTM.insert(i+1, poseTo);
            // update iSAM
            isamTM->update(gtSAMgraphTM, initialEstimateTM);
            isamTM->update();
            gtSAMgraphTM.resize(0);
            initialEstimateTM.clear();
        }

        // Eigen::Affine3f wrongPose = pclPointToAffine3f(temporaryCloudKeyPoses6D->points[tempSize -1 ]);
        gtsam::Pose3 poseCorr = Affine3f2gtsamPose(correctedPose);

        // std::cout<<" add prior factor instead to constrain the covariances"<<endl;
        // cannot put it in the loop above
        // odomFactor needs to be a smooth one!!!
        
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[tempSize - 2 ]);
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[tempSize - 1 ]);
        gtSAMgraphTM.add(BetweenFactor<Pose3>(tempSize - 2 , tempSize -1, poseFrom.between(poseTo), odometryNoise));

        
        gtSAMgraphTM.add(PriorFactor<Pose3>(tempSize - 1, poseCorr, corrNoise));
        initialEstimateTM.insert(tempSize - 1, poseCorr);
        
        // std::cout<<"before opt. "<< poseCorr.translation().x()<<" "<< poseCorr.translation().y()<<" "<< poseCorr.translation().z()<<endl;

        // update iSAM
        isamTM->update(gtSAMgraphTM, initialEstimateTM);
        isamTM->update();
        isamTM->update();
        isamTM->update();
        gtSAMgraphTM.resize(0);
        initialEstimateTM.clear();

        
        isamCurrentEstimateTM = isamTM->calculateEstimate();

        // std::cout<<"pose correction: "<<isamCurrentEstimateTM.size()<<endl;

        std::vector<int> keyPoseSearchIdx;
        std::vector<float> keyPoseSearchDist;
        for (int i = priorNode; i < tempSize; i++)
        {
            // change every loop
            pcl::KdTreeFLANN<PointType>::Ptr keyPosesTree(new pcl::KdTreeFLANN<PointType>());
            
            mtx.lock();
            keyPosesTree->setInputCloud(cloudKeyPoses3D);
            mtx.unlock();

            auto poseCov = isamTM->marginalCovariance(i);

            temporaryCloudKeyPoses3D->points[i].x = isamCurrentEstimateTM.at<Pose3>(i).translation().x();
            temporaryCloudKeyPoses3D->points[i].y = isamCurrentEstimateTM.at<Pose3>(i).translation().y();
            temporaryCloudKeyPoses3D->points[i].z = isamCurrentEstimateTM.at<Pose3>(i).translation().z();

            temporaryCloudKeyPoses6D->points[i].x = temporaryCloudKeyPoses3D->points[i].x;
            temporaryCloudKeyPoses6D->points[i].y = temporaryCloudKeyPoses3D->points[i].y;
            temporaryCloudKeyPoses6D->points[i].z = temporaryCloudKeyPoses3D->points[i].z;
            temporaryCloudKeyPoses6D->points[i].roll  = isamCurrentEstimateTM.at<Pose3>(i).rotation().roll();
            temporaryCloudKeyPoses6D->points[i].pitch = isamCurrentEstimateTM.at<Pose3>(i).rotation().pitch();
            temporaryCloudKeyPoses6D->points[i].yaw   = isamCurrentEstimateTM.at<Pose3>(i).rotation().yaw();

            // temporaryCloudKeyPoses3D->points[i].intensity = i; // no change here actually
            // temporaryCloudKeyPoses6D->points[i].intensity = i;

            keyPosesTree->nearestKSearch(temporaryCloudKeyPoses3D->points[i],1, keyPoseSearchIdx, keyPoseSearchDist);
            // std::cout<<"nearest: "<<keyPoseSearchDist.size()<<endl;
            mtx.lock();
            if (keyPoseSearchDist[0] < 2*surroundingKeyframeDensity)
            {
                // std::cout<<keyPoseSearchIdx[0]<<endl;
                cloudKeyPoses3D->erase(cloudKeyPoses3D->begin() + keyPoseSearchIdx[0]);
                cloudKeyPoses6D->erase(cloudKeyPoses6D->begin() + keyPoseSearchIdx[0]);
                cornerCloudKeyFrames.erase(cornerCloudKeyFrames.begin() + keyPoseSearchIdx[0]);
                surfCloudKeyFrames.erase(surfCloudKeyFrames.begin() + keyPoseSearchIdx[0]);
                isIndoorKeyframe.erase(isIndoorKeyframe.begin() + keyPoseSearchIdx[0]);
            }
            mtx.unlock();
        }

        pcl::PointCloud<PointType>::Ptr cloudLocal(new pcl::PointCloud<PointType>());
        for (int i=0;i<(int)tempSize;i++)
        {
            int idx = temporaryCloudKeyPoses3D->points[i].intensity;
            *cloudLocal += *transformPointCloud(temporarySurfCloudKeyFrames[idx],&temporaryCloudKeyPoses6D->points[i]);
            *cloudLocal += *transformPointCloud(temporaryCornerCloudKeyFrames[idx],&temporaryCloudKeyPoses6D->points[i]);
        }
        publishCloud(&pubMergedMap, cloudLocal, timeLidarInfoStamp, mapFrame);



        for (int i = priorNode; i < tempSize; i++)
        {
            cloudKeyPoses3D->push_back(temporaryCloudKeyPoses3D->points[i]); // no "points." in between!!!
            cloudKeyPoses6D->push_back(temporaryCloudKeyPoses6D->points[i]);
            cornerCloudKeyFrames.push_back(temporaryCornerCloudKeyFrames[i]);
            surfCloudKeyFrames.push_back(temporarySurfCloudKeyFrames[i]); 
            isIndoorKeyframe.push_back(isIndoorKeyframeTMM[i]);
        }
        std::cout<<"map merge takes "<<t_merge.toc()<< " ms"<<endl; // negligible

        // reindexing due to the erasing operation
        for (int i = 0; i < (int) cloudKeyPoses3D->size(); i++)
        {
            cloudKeyPoses3D->points[i].intensity = i;
            cloudKeyPoses6D->points[i].intensity = i;
        }

    }

    void updatePathRELOC(const roll::cloud_infoConstPtr& msgIn){
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = msgIn->header.stamp;
        pose_stamped.header.frame_id = mapFrame;
        pose_stamped.pose.position.x = transformTobeMapped[3];
        pose_stamped.pose.position.y = transformTobeMapped[4];
        pose_stamped.pose.position.z = transformTobeMapped[5];
        tf::Quaternion q = tf::createQuaternionFromRPY(transformTobeMapped[0],transformTobeMapped[1],transformTobeMapped[2]);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();
        globalPath.poses.push_back(pose_stamped);
    }

    /*
    * need online calibration for lidar yaw in ENU frame, so far no implementaion
    */
    void gpsHandler(const sensor_msgs::NavSatFixConstPtr& gpsMsg)
    {
        if (gtAvailable || gpsMsg->status.status != 0 || isnan(gpsMsg->latitude) || isnan(gpsMsg->longitude) || isnan(gpsMsg->altitude)) 
        {
            if(debugMode) std::cout<<"Got 6axis gnss message or bad status: "<<gpsMsg->status.status<<endl;
            return;
        }

        double x,y,z; // ENU, note that in nclt, it it NED
        x = sin(deg2rad(gpsMsg->longitude - longi0))*rew*cos(deg2rad(lati0));
        y = sin(deg2rad(gpsMsg->latitude - lati0))*rns;
        z = gpsMsg->altitude -alti0;

        // gps receiver is highly overlapped with lidar center in xy, so neglecting their extrensics
        Eigen::Affine3f affine_gps_to_enu = pcl::getTransformation(x,y,z,0,0,0);

        Eigen::Affine3f affine_lidar_to_enu = affine_gps_to_enu*affine_lidar_to_gps;
        
        if(debugMode) std::cout<<"after conversion x y z: "<<affine_lidar_to_enu(0,3)<<" "<<affine_lidar_to_enu(1,3)<<" "<<affine_lidar_to_enu(2,3)<<endl;
        nav_msgs::Odometry odomGPS;
        // notice PoseWithCovariance order: # (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis; float64[36] covariance
        // different from GTSAM pose covariance order
        odomGPS.pose.covariance[0] = gpsMsg->position_covariance[0];
        odomGPS.pose.covariance[7] = gpsMsg->position_covariance[4];
        odomGPS.pose.covariance[14] = gpsMsg->position_covariance[8];
        odomGPS.pose.pose.position.x = affine_lidar_to_enu(0,3);
        odomGPS.pose.pose.position.y = affine_lidar_to_enu(1,3);
        odomGPS.pose.pose.position.z = affine_lidar_to_enu(2,3);
        odomGPS.pose.pose.orientation.x = 0;
        odomGPS.pose.pose.orientation.y = 0;
        odomGPS.pose.pose.orientation.z = 0;
        odomGPS.pose.pose.orientation.w = 1.0;
        odomGPS.header.stamp = gpsMsg->header.stamp;
        odomGPS.header.frame_id = "/map";

        mtxGPSqueue.lock();
        gpsQueue.push_back(odomGPS);
        mtxGPSqueue.unlock();

        geometry_msgs::PoseStamped pose_stamped_gps;
        pose_stamped_gps.pose = odomGPS.pose.pose;
        pose_stamped_gps.header = odomGPS.header;

        static long cnt_gps = 0;
        cnt_gps++;
        if(cnt_gps%10 == 0 &&  pubPathGPS.getNumSubscribers() != 0)
        {
            globalPathGPS.poses.push_back(pose_stamped_gps);
            globalPathGPS.header = odomGPS.header;
            pubPathGPS.publish(globalPathGPS);
        }
        
        gpsTrajPub.publish(odomGPS);

        gps_file<<odomGPS.header.stamp.toSec()<<" "<<odomGPS.pose.pose.position.x<<" "<<odomGPS.pose.pose.position.y<<" "<<odomGPS.pose.pose.position.z<<
            " "<<odomGPS.pose.covariance[0]<<" "<<odomGPS.pose.covariance[7]<<" "<<odomGPS.pose.covariance[14]<<"\n";
    }


    void gtHandler(const nav_msgs::Odometry::ConstPtr& gtMsg)
    {
        if (isnan(gtMsg->pose.pose.position.x) || isnan(gtMsg->pose.pose.position.y) || isnan(gtMsg->pose.pose.position.z)
        || gtMsg->pose.covariance[0] > 1.0) 
        {
            if(debugMode) std::cout<<"x: "<< gtMsg->pose.pose.position.x<<" cov: "<<gtMsg->pose.covariance[0]<<endl;
            return;
        }
        Eigen::Affine3f affine_sth_to_map,affine_lidar_to_map;
        odometryMsgToAffine3f(*gtMsg, affine_sth_to_map);
        
        //for lab datasets, gt is trans_gps_to_enu, we use enu as map
        if(bodyCentered == false)
        {
            affine_lidar_to_map = affine_sth_to_map*affine_lidar_to_gps;
        }
        else
        {
            // for nclt, gt is trans_body_to_ned, we use NED as map
            affine_lidar_to_map = affine_sth_to_map*affine_lidar_to_body;
        }

        float roll,pitch,yaw,x,y,z;
        // no need for z roll pitch
        // if they are really needed, try set lidar height as the initial height
        pcl::getTranslationAndEulerAngles(affine_lidar_to_map,x,y,z,roll,pitch,yaw);
        tf::Quaternion q = tf::createQuaternionFromRPY(roll,pitch,yaw);
        nav_msgs::Odometry odomGPS;
        odomGPS.header.frame_id = mapFrame;
        odomGPS.header.stamp = gtMsg->header.stamp;
        odomGPS.pose.covariance = gtMsg->pose.covariance;
        odomGPS.pose.pose.position.x = x;
        odomGPS.pose.pose.position.y = y;
        odomGPS.pose.pose.position.z = z;
        odomGPS.pose.pose.orientation.x = q.x();
        odomGPS.pose.pose.orientation.y = q.y();
        odomGPS.pose.pose.orientation.z = q.z();
        odomGPS.pose.pose.orientation.w = q.w();
        mtxGPSqueue.lock();
        gpsQueue.push_back(odomGPS);
        mtxGPSqueue.unlock();
        gpsTrajPub.publish(odomGPS);

        geometry_msgs::PoseStamped pose_stamped_gps;
        pose_stamped_gps.pose = odomGPS.pose.pose;
        pose_stamped_gps.header = odomGPS.header;

        static long cnt_gps = 0;
        cnt_gps++;
        if(cnt_gps%10 == 0 &&  pubPathGPS.getNumSubscribers() != 0)
        {
            globalPathGPS.poses.push_back(pose_stamped_gps);
            globalPathGPS.header = odomGPS.header;
            pubPathGPS.publish(globalPathGPS);
        }
        
    }


    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        // #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    gtsam::Pose3 Affine3f2gtsamPose(Eigen::Affine3f aff){
        float transformOut[6] = {0};
        Affine3f2Trans(aff, transformOut);
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformOut[0], transformOut[1], transformOut[2]), 
                                  gtsam::Point3(transformOut[3], transformOut[4], transformOut[5]));
    }
    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }
    
    bool saveMapService(roll::save_mapRequest& req, roll::save_mapResponse& res)
    {        
        float resMap,resPoseIndoor,resPoseOutdoor,overlapThre;
        
        resMap = req.resolutionMap==0?0.1:req.resolutionMap;
        resPoseIndoor = req.resPoseIndoor;
        resPoseOutdoor = req.resPoseOutdoor;
        overlapThre = req.overlapThre;
        
        if(bashRun){
            // used in IROS
            resMap = 0.1;  //0.1
            resPoseIndoor = 2.0;  // change it back to 5.0/2.0 after done         0.5
            resPoseOutdoor = 5.0; //0.5
            overlapThre = 0.8;
        }
        std::cout << "resMap: " << resMap << " resPoseIndoor: " << resPoseIndoor
             << " resPoseOutdoor: " << resPoseOutdoor
             << " overlapThre: " << overlapThre << endl;
        std::cout << "Average time consumed by mapping is :" << mappingTime << " ms"
             << endl;
        // if (localizationMode) std::cout<<"Times of entering TMM is :"<<TMMcount<<endl;

        // saving pose estimates and GPS signals
        if (savePose && globalPathFusion.poses.empty() == false) 
        {
            // ofstream pose_file;
            std::cout<<"Recording trajectory..."<<endl;

            mtx.lock();
            // higher frequency odometry
            ofstream pose_file3;

            string fileName3;
            if(bashRun) fileName3 = log_dir+"/poses/pose"+ logTime + ".txt";
            else fileName3 = saveKeyframeMapDirectory + "/../dt_poses_semi.txt";
            pose_file3.open(fileName3,ios::out);
            pose_file3.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
            pose_file3.precision(6); // 固定小数位6
            int pointN3 = (int)globalPathFusion.poses.size();
            std::cout<<"fusion pose size: "<<pointN3<<endl;
            for (int i = 0; i < pointN3; ++i)
            {
                geometry_msgs::PoseStamped tmp = globalPathFusion.poses[i];
                double r,p,y;
                tf::Quaternion q(tmp.pose.orientation.x,tmp.pose.orientation.y, tmp.pose.orientation.z,tmp.pose.orientation.w);
                tf::Matrix3x3(q).getRPY(r,p,y);
                // save it in nano sec to compare it with the nclt gt
                pose_file3<<tmp.header.stamp.toSec()*1e+6<<" "<<tmp.pose.position.x<<" "<<tmp.pose.position.y<<" "<<
                tmp.pose.position.z<<" "<<r<<" "<<p<<" "<<y<<" "<<"\n";
            }
            pose_file3.close();
            mtx.unlock();

            mtx.lock();
            // higher frequency odometry
            ofstream pose_file4;
            string fileName4;
            if(bashRun) fileName4 = log_dir+"/fusion_poses/pose"+ logTime + ".txt";
            else fileName4 = saveKeyframeMapDirectory + "/../fusion_poses.txt";
            pose_file4.open(fileName4,ios::out);
            pose_file4.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
            pose_file4.precision(6); // 固定小数位6
            int pointN4 = (int)globalPathFusionVINS.poses.size();
            std::cout<<"vinsfusion pose size: "<<pointN4<<endl;
            for (int i = 0; i < pointN4; ++i)
            {
                geometry_msgs::PoseStamped tmp = globalPathFusionVINS.poses[i];
                double r,p,y;
                tf::Quaternion q(tmp.pose.orientation.x,tmp.pose.orientation.y, tmp.pose.orientation.z,tmp.pose.orientation.w);
                tf::Matrix3x3(q).getRPY(r,p,y);
                // save it in nano sec to compare it with the nclt gt
                pose_file4<<tmp.header.stamp.toSec()*1e+6<<" "<<tmp.pose.position.x<<" "<<tmp.pose.position.y<<" "<<
                tmp.pose.position.z<<" "<<r<<" "<<p<<" "<<y<<" "<<"\n";
            }
            pose_file4.close();
            std::cout<<"Trajectory recording finished!"<<endl;
            mtx.unlock();
        }


        // save keyframe map: for every keyframe, save keyframe pose, edge point pcd, surface point pcd
        // keyframe pose in one file
        // every keyframe has two other files: cornerI.pcd surfI.pcd
        if (savePCD && cloudKeyPoses3D->empty() == false)
        {
            std::cout << "****************************************************" << endl;
            std::cout << "keyframe destination: " << saveKeyframeMapDirectory << endl;

            // save key frame transformations
            pcl::io::savePCDFileBinary(saveKeyframeMapDirectory + "/../trajectory.pcd", *cloudKeyPoses3D);
            pcl::io::savePCDFileBinary(saveKeyframeMapDirectory + "/../transformations.pcd", *cloudKeyPoses6D);
            // extract global point cloud map
            pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloudDS(new pcl::PointCloud<PointType>());
            for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
                int idx = cloudKeyPoses3D->points[i].intensity;
                *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[idx],  &cloudKeyPoses6D->points[idx]);
                *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[idx],    &cloudKeyPoses6D->points[idx]);
                // std::cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...\n";
            }

            std::cout << "\nSave resolution: " << resMap << endl;
            // down-sample and save corner cloud
            downSizeFilterCorner.setInputCloud(globalCornerCloud);
            downSizeFilterCorner.setLeafSize(resMap, resMap, resMap);
            downSizeFilterCorner.filter(*globalCornerCloudDS);
            pcl::io::savePCDFileBinary(saveKeyframeMapDirectory + "/../CornerMap.pcd", *globalCornerCloudDS);
            // down-sample and save surf cloud
            downSizeFilterSurf.setInputCloud(globalSurfCloud);
            downSizeFilterSurf.setLeafSize(resMap, resMap, resMap);
            downSizeFilterSurf.filter(*globalSurfCloudDS);
            pcl::io::savePCDFileBinary(saveKeyframeMapDirectory + "/../SurfMap.pcd", *globalSurfCloudDS);

            // save global point cloud map
            *globalMapCloud += *globalCornerCloud;
            *globalMapCloud += *globalSurfCloud;
            downSizeFilterCorner.setInputCloud(globalMapCloud);
            downSizeFilterCorner.setLeafSize(resMap, resMap, resMap);
            downSizeFilterCorner.filter(*globalMapCloudDS);
            int ret = pcl::io::savePCDFileBinary(saveKeyframeMapDirectory + "/../GlobalMap.pcd", *globalMapCloudDS);
            res.success = ret == 0;

            std::cout << "Saving map to pcd files completed\n" << endl;
        }
        
        if(saveKeyframeMap && cloudKeyPoses6D->empty() == false ){ 
            int keyframeN = (int)cloudKeyPoses6D->size();

            if(bashRun)    saveKeyframeMapDirectory = map_dir + "/keyframes";

            std::cout<<"Map Saving to "+saveKeyframeMapDirectory<<endl;
            ofstream pose_file;
            pose_file.open(saveKeyframeMapDirectory+"/poses.txt",ios::out); // downsampled
            
            if(!pose_file.is_open())
            {
                std::cout<<"Cannot open"<<saveKeyframeMapDirectory+"/poses.txt"<<std::endl;
                return false;
            }
            std::cout<<"There are "<<keyframeN<<" keyframes before downsampling"<<endl;

            std::cout<<"********************Saving keyframes and poses one by one**************************"<<endl;
            pcl::PointCloud<PointType>::Ptr cloudKeyPoses3DDS(new pcl::PointCloud<PointType>());
            keyframeSparsification(cloudKeyPoses3DDS,resMap,resPoseIndoor,resPoseOutdoor,overlapThre);

            int keyframeNDS = cloudKeyPoses3DDS->size();
            std::cout << "There are " << keyframeNDS << " keyframes after downsampling" << endl;
  
            std::vector<int> keyframeSearchIdx;
            std::vector<float> keyframeSearchDist;
            pcl::KdTreeFLANN<PointType>::Ptr kdtreeKeyframes(new pcl::KdTreeFLANN<PointType>());
            kdtreeKeyframes->setInputCloud(cloudKeyPoses3D);
            int i = 0;            
            // recover downsampled intensities
            for(auto& pt:cloudKeyPoses3DDS->points)
            {                
                kdtreeKeyframes->nearestKSearch(pt,1,keyframeSearchIdx,keyframeSearchDist); 
                pt.intensity = cloudKeyPoses6D->points[keyframeSearchIdx[0]].intensity;  
                pcl::io::savePCDFileBinary(saveKeyframeMapDirectory + "/corner" + std::to_string(i)+".pcd", *cornerCloudKeyFrames[pt.intensity]);
                pcl::io::savePCDFileBinary(saveKeyframeMapDirectory + "/surf" + std::to_string(i)+".pcd", *surfCloudKeyFrames[pt.intensity]);
                pose_file<<setiosflags(ios::fixed)<<setprecision(6)<<cloudKeyPoses6D->points[pt.intensity].time
                <<" "<<cloudKeyPoses6D->points[pt.intensity].x<<" "<<cloudKeyPoses6D->points[pt.intensity].y<<" "<<cloudKeyPoses6D->points[pt.intensity].z
                <<" "<<cloudKeyPoses6D->points[pt.intensity].roll<<" "<<cloudKeyPoses6D->points[pt.intensity].pitch<<" "<<cloudKeyPoses6D->points[pt.intensity].yaw
                << " " << i<<" "<<isIndoorKeyframe[pt.intensity]<<"\n";
                i++;
                if((i+1)%100 == 0) std::cout<<i<<" keyframes saved!"<<endl;
            }
            pose_file.close();
            std::cout<<"Keyframes Saving Finished!"<<endl;

        }
        res.success = true;
        doneSavingMap = true;
        return true;
    }

   
    void keyframeSparsification(pcl::PointCloud<PointType>::Ptr  &cloudKeyPoses3DDS, float resMap, float resPoseIndoor, float resPoseOutdoor,
                                float overlapThre)
    {
        if(resPoseIndoor == 0 || resPoseOutdoor == 0 || overlapThre == 0) 
        {
            pcl::copyPointCloud(*cloudKeyPoses3D,*cloudKeyPoses3DDS);
            std::cout << "No downsampling performed on kefyrame map saving" << endl;
            ROS_WARN("No downsampling performed on kefyrame map saving");
            return;
        }
        TicToc sparsiTime;
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalKeyPoses(new pcl::KdTreeFLANN<PointType>());
        kdtreeGlobalKeyPoses->setInputCloud(cloudKeyPoses3D);

        // separate indoor or outdoor, sparsify crudely
        pcl::PointCloud<PointType>::Ptr keyPosesIndoor(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr keyPosesOutdoor(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr keyPosesIndoorDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr keyPosesOutdoorDS(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->points.size();i++)
        {
            if (isIndoorKeyframe[cloudKeyPoses3D->points[i].intensity] == 1) 
                keyPosesIndoor->push_back(cloudKeyPoses3D->points[i]);
            else 
                keyPosesOutdoor->push_back(cloudKeyPoses3D->points[i]);
        }

        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;      
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPosesI;

        //indoor
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPosesO;
        downSizeFilterGlobalMapKeyPosesO.setLeafSize(resPoseIndoor,resPoseIndoor,resPoseIndoor);
        downSizeFilterGlobalMapKeyPosesO.setInputCloud(keyPosesIndoor);
        downSizeFilterGlobalMapKeyPosesO.filter(*keyPosesIndoorDS);

        // fix the keyframe downsample bug, keep intensity as an index of adding sequence
        for(auto& pt : keyPosesIndoorDS->points)
        {
            kdtreeGlobalKeyPoses->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
            cloudKeyPoses3DDS->push_back(pt);
        }
        std::cout<<"indoor: "<<keyPosesIndoorDS->size()<<" frames" <<endl;
        //outdoor
        downSizeFilterGlobalMapKeyPosesI.setLeafSize(resPoseOutdoor,resPoseOutdoor,resPoseOutdoor);
        downSizeFilterGlobalMapKeyPosesI.setInputCloud(keyPosesOutdoor);
        downSizeFilterGlobalMapKeyPosesI.filter(*keyPosesOutdoorDS);
        // fix the keyframe downsample bug
        for(auto& pt : keyPosesOutdoorDS->points)
        {
            kdtreeGlobalKeyPoses->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
            cloudKeyPoses3DDS->push_back(pt);
        }
         std::cout<<"outdoor: "<<keyPosesOutdoorDS->size()<<" frames" <<endl;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rateMap(0.2);        
        while (ros::ok())
        {
            publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLidarInfoStamp, mapFrame);           
            publishGlobalMap();
            rateMap.sleep();
        }

        if (savePCD == false && saveKeyframeMap == false && savePose == false)
            return;

        roll::save_mapRequest  req;
        roll::save_mapResponse res;

        if (!doneSavingMap)
        {
            if(!saveMapService(req, res))   std::cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLidarCloudSurround.getNumSubscribers() == 0 || cloudKeyPoses3D->empty() == true)
        {
            return;
        }
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        PointType pt;
        pt.x = affine_imu_to_map(0,3);
        pt.y = affine_imu_to_map(1,3);
        pt.z = affine_imu_to_map(2,3);
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(pt, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key poses 
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        // fix the keyframe downsample bug
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        // only for visualization
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
        {
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points: why it is not working
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS); 
        // std::cout<<globalMapKeyFramesDS->size()<<endl;       
        publishCloud(&pubLidarCloudSurround, globalMapKeyFramesDS, timeLidarInfoStamp, mapFrame);

    }

    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false )
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            if (!localizationMode) 
            {
                performLoopClosure();
                visualizeLoopClosure();
            }
            
        }
    }


    void performLoopClosure()
    {
        if (cloudKeyPoses3D->empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
            return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLidarInfoStamp, mapFrame);

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> registration;
        registration.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        registration.setMaximumIterations(100);
        registration.setTransformationEpsilon(1e-6);
        registration.setEuclideanFitnessEpsilon(1e-6);
        registration.setRANSACIterations(0);

        // // gicp is way better than icp
        // static pcl::GeneralizedIterativeClosestPoint<PointType,PointType> registration;

        // Align clouds
        registration.setInputSource(cureKeyframeCloud);
        registration.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        registration.align(*unused_result);

        fitnessScore = registration.getFitnessScore();

        if (registration.hasConverged() == false ||  fitnessScore > historyKeyframeFitnessScore)
            return;
        
        if(debugMode) std::cout<<"found loop, fitness score: "<< fitnessScore<<endl;
        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, registration.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLidarInfoStamp, mapFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = registration.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);

        Vector6 << fitnessScore, fitnessScore, fitnessScore, fitnessScore, fitnessScore, fitnessScore;
        // noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(corrNoise);
        noiseVec.push_back(fitnessScore);
        mtx.unlock();

        // add loop constriant
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        // reset before usage!
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        // std::cout<<copy_cloudKeyPoses6D->points[id].time-cloudKeyPoses6D->points[id].time<<" "<<copy_cloudKeyPoses6D->points[id].time-cloudInfoTime <<endl;
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            
            if (abs(copy_cloudKeyPoses6D->points[id].time - cloudInfoTime) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum, bool doFiltering = true)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            // it may happen in temporary mapping mode
            float dist = pointDistance(copy_cloudKeyPoses3D->points[key],copy_cloudKeyPoses3D->points[keyNear]);
            if (  dist > surroundingKeyframeSearchRadius)
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);  
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        if (doFiltering){
            pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
            downSizeFilterICP.setInputCloud(nearKeyframes);
            downSizeFilterICP.filter(*cloud_temp);
            *nearKeyframes = *cloud_temp;
        }
    }
    // copied to the original, so that reloc part can use this func
    void loopFindNearKeyframesReloc(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum, bool doFiltering = true)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
                // copied to the original, so that reloc part can use this func too
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &cloudKeyPoses6D->points[keyNear]);  
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        if (doFiltering){
            pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
            downSizeFilterICP.setInputCloud(nearKeyframes);
            downSizeFilterICP.filter(*cloud_temp);
            *nearKeyframes = *cloud_temp;
        }
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = mapFrame;
        markerNode.header.stamp = timeLidarInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = mapFrame;
        markerEdge.header.stamp = timeLidarInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
        // ROS_INFO_STREAM("Loop noise: "<<noiseVec[noiseVec.size()-1]);

    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, Eigen::Affine3f transCur)
    {
        int cloudSize = cloudIn->size();
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        cloudOut->resize(cloudSize);
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    void initialpose_callback(const rvizPoseType& pose_msg) {
        ROS_INFO("initial pose received!!");
        std::lock_guard<std::mutex> lock(pose_estimator_mutex);
        poseEstVec.push_back(pose_msg);
        if (relocByRviz()){
            ROS_INFO("Got pose estimate");
            poseGuessFromRvizAvailable = true;
        }
    }


    bool relocByRviz() {
        if (!poseEstVec.empty())
        {
            // use mutex for copy process to prevent concurrent change to the cloudKeyPoses3D,seems no need
            Eigen::Affine3f transBackup;
            mtx.lock(); 
            transBackup = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
            mtx.unlock();
            kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            PointType pt;
            vector<int> pointSearchIndLoop;
            vector<float> pointSearchSqDisLoop;
            rvizPoseType pose_msg = poseEstVec.back();
            const auto& p = pose_msg->pose.pose.position;
            const auto& q = pose_msg->pose.pose.orientation;
            tf::Quaternion q_tf(q.x,q.y,q.z,q.w);
            double roll,pitch,yaw;
            tf::Matrix3x3(q_tf).getRPY(roll,pitch,yaw);
            Eigen::Affine3f affineGuess = pcl::getTransformation(p.x, p.y, p.z, (float)roll,(float)pitch,(float)yaw);
            relocCorrection = affineGuess*transBackup.inverse(); 
            poseEstVec.clear();
            return true;     
        }
        return false;
    }

    void updateInitialGuess()
    {
        
        // // only use rviz in loc mode
        // if(localizationMode && poseGuessFromRvizAvailable)
        // {
        //     Eigen::Affine3f tWrong = trans2Affine3f(transformTobeMapped);
        //     Eigen::Affine3f tCorrect = relocCorrection*tWrong;

        //     affine_imu_to_map = tCorrect;
        //     affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse(); // for real-time odometry publish
        //     pcl::getTranslationAndEulerAngles(tCorrect, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
        //                                                     transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        //     for(int i = 0;i < 6;i++){
        //         rvizGuess[i] = transformTobeMapped[i];
        //         transformBeforeMapped[i] = transformTobeMapped[i];
        //     }
        //     printTrans("The reloc pose given by rviz: ",transformTobeMapped);
        //     poseGuessFromRvizAvailable = false;
        //     tryReloc = true;
        //     return;
        // }

        // initialization 
        if (relocSuccess == false)
        {
            if(!gtAvailable && !useGPS)
            {
                // by default, the 1st localized frame pose is identity(), in fastlio, actually no
                for(int i=0;i<6;i++){
                    transformTobeMapped[i] = initialGuess[i];
                    transformBeforeMapped[i] = initialGuess[i];
                }
                affine_imu_to_map = trans2Affine3f(transformTobeMapped);
                affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse();
                std::cout<<"Todom2map: "<<affine_odom_to_map.matrix()<<endl;
                printTrans("Reloc success by zero initial: ",transformTobeMapped);
                global_estimator->setTgl(affine_odom_to_map.matrix().cast<double>());
                global_estimator->inputGlobalLocPose(cloudInfoTime, affine_imu_to_map.matrix().cast<double>(), 0.1, 4e-4);   
                // std::cout<<"Relocalized without ground truth or GPS!"<<endl;
                relocSuccess = true;
                return ;
            }
            //  use ground truth for initialization in both modes!
            if(!gpsQueue.empty() && gtAvailable)
            {
                // works when gps odometry is more often than lidar odometry
                
                while(!gpsQueue.empty() && cloudInfoTime - gpsQueue.front().header.stamp.toSec()  > 0.1 )
                {
                    // std::cout<<setiosflags(ios::fixed)<<setprecision(3)<<gpsQueue.front().header.stamp.toSec() - cloudInfoTime<<endl;
                    mtxGPSqueue.lock();
                    gpsQueue.pop_front();
                    mtxGPSqueue.unlock();
                }                  

                if(gpsQueue.empty() || cloudInfoTime - gpsQueue.front().header.stamp.toSec() < -0.1) return;
                odometryMsgToAffine3f(gpsQueue.front(),affine_imu_to_map);
                Affine3f2Trans(affine_imu_to_map,transformTobeMapped);
                affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse();
                std::cout<<"Todom2map: "<<affine_odom_to_map.matrix()<<endl;
                for(int i=0;i<6;i++)
                {
                    transformBeforeMapped[i] = transformTobeMapped[i];
                }
                printTrans("Reloc success by RTK with heading, Initial: ",transformTobeMapped);
                global_estimator->setTgl(affine_odom_to_map.matrix().cast<double>());
                global_estimator->inputGlobalLocPose(cloudInfoTime, affine_imu_to_map.matrix().cast<double>(), 0.1, 4e-4);   
                relocSuccess = true; 
                return;
            }
            if(!gpsQueue.empty() && useGPS )
            {
                while(!gpsQueue.empty() && cloudInfoTime -gpsQueue.front().header.stamp.toSec() > 0.1 )
                {
                    mtxGPSqueue.lock();
                    gpsQueue.pop_front();
                    mtxGPSqueue.unlock();
                }
                if(gpsQueue.empty() || cloudInfoTime - gpsQueue.front().header.stamp.toSec() < -0.1) return;
                Eigen::Affine3f affine_gnss;
                odometryMsgToAffine3f(gpsQueue.front(),affine_gnss);
                gpsQueue.pop_front();
                gnss_lidar_initializer->getSyncFrames(cloudInfoTime,affine_imu_to_odom,affine_gnss);            
                if(gnss_lidar_initializer->tryInitializing() == true)
                {
                    affine_odom_to_map = gnss_lidar_initializer->getTer();                 
                    affine_imu_to_map = affine_odom_to_map*affine_imu_to_odom;
                    Affine3f2Trans(affine_imu_to_map,transformTobeMapped);
                    std::cout<<"Todom2map: "<<affine_odom_to_map.matrix()<<endl;
                    for(int i=0;i<6;i++)
                    {
                        transformBeforeMapped[i] = transformTobeMapped[i];
                    }
                    global_estimator->setTgl(affine_odom_to_map.matrix().cast<double>());
                    global_estimator->inputGlobalLocPose(cloudInfoTime, affine_imu_to_map.matrix().cast<double>(), 0.1, 4e-4);  
                    relocSuccess = true;
                }
            }
            // else if (count < tryRuns) 
            // {
            //     count++;
            //     ROS_INFO("Initializing: %d",count);
            // }
            // else
            // {
            //     // tired of waiting for gt, try initial guess given. 
            //     ROS_WARN("Waiting for correct initial guess, probably need rviz for reloc");
            //     for(int i=0;i<6;i++){
            //         transformTobeMapped[i] = initialGuess[i];
            //         transformBeforeMapped[i] = initialGuess[i];
            //     }
            //     Eigen::Affine3f affine_body_to_map = trans2Affine3f(transformTobeMapped);
            //     affine_imu_to_map = affine_body_to_map*affine_imu_to_body;
            //     affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse();
            //     printTrans("Initial: ",transformTobeMapped); //no more waiting for rviz guess 
            //     global_estimator->setTgl(affine_odom_to_map.matrix().cast<double>());     
            //     global_estimator->inputGlobalLocPose(cloudInfoTime, affine_imu_to_map.matrix().cast<double>(), 0.5, 0.1);              
            // }       
        }
        
        float arrayTmp[6];
        affine_imu_to_map = affine_odom_to_map*affine_imu_to_odom;
        Affine3f2Trans(affine_imu_to_map, arrayTmp);
        
        for (int i=0;i<6;i++)                
        {
            transformTobeMapped[i] = arrayTmp[i];
            transformBeforeMapped[i] = transformTobeMapped[i];
        }
        return;
    }

    void extractNearby()
    {
        if (cloudKeyPoses3D->empty() == true) 
            return;
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        PointType pt;
        pt.x=transformTobeMapped[3];
        pt.y=transformTobeMapped[4];
        pt.z=transformTobeMapped[5];
        
        kdtreeSurroundingKeyPoses->radiusSearch(pt, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);

        if (pointSearchInd.empty()) 
        {
            // std::cout<<pt.x<< " "<<pt.y <<endl;
            // ROS_WARN("No nearby keyposes within %f meters",surroundingKeyframeSearchRadius);
            return;
        }

        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // downsampling is important especially at places where trajectories overlap when doing slam
        
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        for(auto& pt : surroundingKeyPosesDS->points) // recover the intensity field averaged by voxel filter
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        if (!localizationMode) 
        {   // seems useless --- commented by ACE 2024.4.9
            // also extract some latest key frames in case the robot rotates in one position
            // more recent ones matches better with current frames
            int numPoses = cloudKeyPoses3D->size();
            for (int i = numPoses-1; i >= 0; --i)
            {
                if (cloudInfoTime - cloudKeyPoses6D->points[i].time < 10.0)
                    surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
                else
                    break;
            }
        }
        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        lidarCloudCornerFromMap->clear();
        lidarCloudSurfFromMap->clear(); 
        // TicToc transPC;
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;

            *lidarCloudCornerFromMap += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *lidarCloudSurfFromMap   += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);    
        }
        
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(lidarCloudCornerFromMap);
        downSizeFilterCorner.filter(*lidarCloudCornerFromMapDS);
        lidarCloudCornerFromMapDSNum = lidarCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(lidarCloudSurfFromMap);
        downSizeFilterSurf.filter(*lidarCloudSurfFromMapDS);
        lidarCloudSurfFromMapDSNum = lidarCloudSurfFromMapDS->size();
    }

    // Added by ACE 2024.4.9
    void publishLocalMapforSharedControl() {
      if (cloudKeyPoses3D->empty() == true)
        return;
    //   std::cout << "cloudKeyPoses3D size: " << cloudKeyPoses3D->size() << endl;
      pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());
      kdtree->setInputCloud(cloudKeyPoses3D);  // create kd-tree
      PointType pt;
      pt.x = transformTobeMapped[3];
      pt.y = transformTobeMapped[4];
      pt.z = transformTobeMapped[5];

      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;
      kdtree->radiusSearch(
          pt, (double)surroundingKeyframeSearchRadiusforSharedControl,
          pointSearchInd, pointSearchSqDis);
    //   std::cout << "current pose: " << pt.x << " " << pt.y << " " << pt.z << endl;
    //   std::cout << "pointSearchInd size: " << pointSearchInd.size() << endl;
      if (pointSearchInd.empty()) {
        return;
      }

      pcl::PointCloud<PointType>::Ptr cloudToExtractforSharedControl(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)pointSearchInd.size(); ++i) {
        // 如果时间差大于一定值 并且 当前关键帧和周围历史关键帧的序号差大于一定值（即不考虑走了很远又回到附近时的历史关键帧，减弱时间变换的干扰）
        // 如果只判断时间差，在轮椅一直静止时会有问题
        int id = pointSearchInd[i];
        // if ((cloudInfoTime - cloudKeyPoses6D->points[id].time > 20.0) &&
        //     (cloudKeyPoses3D->back().intensity - cloudKeyPoses3D->points[id].intensity > 10))
        if (cloudInfoTime - cloudKeyPoses6D->points[id].time > 5.0)
          continue;
        cloudToExtractforSharedControl->push_back(cloudKeyPoses3D->points[id]);
      }
    //   std::cout << "cloudToExtractforSharedControl size: " << cloudToExtractforSharedControl->size() << endl;

      pcl::PointCloud<PointType>::Ptr lidarCloudCornerFromMapforSharedControl(new pcl::PointCloud<PointType>());;
      pcl::PointCloud<PointType>::Ptr lidarCloudSurfFromMapforSharedControl(new pcl::PointCloud<PointType>());;
      for (int i = 0; i < (int)cloudToExtractforSharedControl->size(); ++i) {
        int thisKeyInd = (int)cloudToExtractforSharedControl->points[i].intensity;
        //maybe not clear
        *lidarCloudCornerFromMapforSharedControl +=
            *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                                 &cloudKeyPoses6D->points[thisKeyInd]);
        *lidarCloudSurfFromMapforSharedControl +=
            *transformPointCloud(surfCloudKeyFrames[thisKeyInd],
                                 &cloudKeyPoses6D->points[thisKeyInd]);
      }

      pcl::PointCloud<PointType>::Ptr cloudLocalforSharedControl(new pcl::PointCloud<PointType>());
      *cloudLocalforSharedControl += *lidarCloudCornerFromMapforSharedControl;
      *cloudLocalforSharedControl += *lidarCloudSurfFromMapforSharedControl;
    //   std::cout << "cloudLocalforSharedControl size: " << cloudLocalforSharedControl->size() << endl << endl;

      // transformed to LiDAR frame
      *cloudLocalforSharedControl = *transformPointCloud(cloudLocalforSharedControl, trans2Affine3f(transformTobeMapped).inverse());

      // crop the local map within a box
      pcl::PassThrough<PointType> range_pass;
      range_pass.setInputCloud(cloudLocalforSharedControl);
      range_pass.setFilterFieldName("x");
      range_pass.setFilterLimits(-surroundingLocalMapCropRadiusforSharedControl, surroundingLocalMapCropRadiusforSharedControl);
      range_pass.filter(*cloudLocalforSharedControl);
      range_pass.setInputCloud(cloudLocalforSharedControl);
      range_pass.setFilterFieldName("y");
      range_pass.setFilterLimits(-surroundingLocalMapCropRadiusforSharedControl, surroundingLocalMapCropRadiusforSharedControl);
      range_pass.filter(*cloudLocalforSharedControl);

      *cloudLocalforSharedControl += *lidarCloudRaw;
      *cloudLocalforSharedControl += *rslidar_raw;
      publishCloud(&pubRecentKeyFramesforSharedControl, cloudLocalforSharedControl, timeLidarInfoStamp, "rslidar");
    }

    void downsampleCurrentScan()
    {
        if (cloudKeyPoses3D->empty() )
            return;
        // Downsample cloud from current scan
        lidarCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(lidarCloudCornerLast);
        downSizeFilterCorner.filter(*lidarCloudCornerLastDS);
        lidarCloudCornerLastDSNum = lidarCloudCornerLastDS->size();
        // std::cout<<"After sampling: "<<lidarCloudCornerLastDSNum<<endl;
        lidarCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(lidarCloudSurfLast);
        downSizeFilterSurf.filter(*lidarCloudSurfLastDS);
        lidarCloudSurfLastDSNum = lidarCloudSurfLastDS->size();
    }

    void scan2MapOptimization()
    {
        // no clouds nearby
        if (cloudKeyPoses3D->empty() || lidarCloudCornerFromMapDS->empty() || lidarCloudSurfFromMapDS->empty())
            return;
        
        // std::cout<<"corner, surf points: "<<lidarCloudCornerLastDSNum<<" "<<lidarCloudSurfLastDSNum<<endl;
        if (lidarCloudCornerLastDSNum > edgeFeatureMinValidNum && lidarCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // get a copy of those clouds might lower down speed
            LOAMmapping LM(lidarCloudCornerLastDS, lidarCloudSurfLastDS, lidarCloudCornerFromMapDS, lidarCloudSurfFromMapDS, affine_imu_to_map);
            LM.match();
            
            // for relocalization in loc mode: only needed when used in actual world
            if (LM.inlier_ratio2 > 0.3 && tryReloc == true)
            {
                ROS_INFO_STREAM("At time "<< cloudInfoTime - rosTimeStart <<" sec, relocalization succeeds!");
                relocSuccess = true;
                tryReloc = false;
                transformUpdate(); // for giving fusion pose immediate map_odom
            }

            // if (relocSuccess == true && localizationMode == true)
            // {
            //     // for entering TMM: 
            //     //inlier_ratio2 is more sensitive to map extension, and will suffice
            //     if ( LM.inlier_ratio2 < startTemporaryMappingInlierRatioThre && temporaryMappingMode == false)
            //     {
            //         ROS_INFO_STREAM("At time "<< cloudInfoTime - rosTimeStart <<" sec, Entering temporary mapping mode due to poor mapping performace");
            //         ROS_INFO_STREAM("Inlier ratio2: "<< LM.inlier_ratio2);                        
            //         temporaryMappingMode = true; // here is the case for outdated map
            //         startTemporaryMappingIndex = temporaryCloudKeyPoses3D->size();
            //         frameTobeAbandoned = true;
            //         TMMcount++;                   
            //     }
                
            //     // more strict to exit TMM for map updating
            //     if (LM.inlier_ratio2 > exitTemporaryMappingInlierRatioThre && int(temporaryCloudKeyPoses3D->size()) > slidingWindowSize + 10 && temporaryMappingMode == true)
            //     {
            //         correctedPose = LM.affine_out;// notice: the correction cannot be simply the correction for last keyframe!
            //         affine_imu_to_map = LM.affine_out;
            //         // LM.getTransformation(transformTobeMapped); // don't change it here, need original one as odom factor
            //         goodToMergeMap = true;
            //         std::cout<<"Now it is okay to merge the temporary map"<<endl;
            //     }
            // }
            // only correct pose from fastlio when mapping quality is good
            // if (temporaryMappingMode == false && LM.isDegenerate == false)  // worse
            // when building a map, it is better to fuse LIO and global matching
            if (temporaryMappingMode == false || localizationMode == false) 
            {
                           
                // fusion with ceres: currently only for smoothing, not for pose guess.
                // cannot update Tgl immediately because opt takes time, getting a delayed Tgl is rather forfeiting it

                if (goodToMergeMap) // reset for the frame of merging
                    global_estimator->resetOptimization(LM.affine_out.matrix().cast<double>());
                else 
                    // radian error is proportional to meter, 1/250
                    global_estimator->inputGlobalLocPose(cloudInfoTime, LM.affine_out.matrix().cast<double>(), 0.1, 4e-4);             
                
                affine_imu_to_map = LM.affine_out;
                Affine3f2Trans(affine_imu_to_map,transformTobeMapped);
                // printTrans("trans: ",transformTobeMapped);
            }
     
            if(saveLog)
            {
                // for recording mapping logs
                mtx.lock(); // need lock for fitnessScore
                pose_log_file<<setw(20)<<cloudInfoTime<<" "<<transformTobeMapped[0]<<" "<<transformTobeMapped[1]<<" "
                <<transformTobeMapped[2]<<" "<<transformTobeMapped[3]<<" "<<transformTobeMapped[4]<<" "<<
                transformTobeMapped[5]<<" "<< LM.inlier_ratio<<" "<<LM.inlier_ratio2<<" "<<LM.regiError<<
                " "<<temporaryMappingMode<<" "<<mappingTime<<" "<<fitnessScore<<
                " "<<cpu_usage_ratio<< " "<<memory_usage<< "\n";
                mtx.unlock();
            }      
            // std::cout<<"inlierRatio2: "<<LM.inlier_ratio2<<endl;        
        } 
        else 
        {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", lidarCloudCornerLastDSNum, lidarCloudSurfLastDSNum);
        }
    }

    void resetISAM(){
        ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        isam = new ISAM2(optParameters); // why it is okay for liosam not to add "new"?

        gtsam::NonlinearFactorGraph newGraphFactors;
        gtSAMgraph = newGraphFactors;
    }

    int saveFrame()
    {
        // indoor 1; outdoor 0; not a keyframe -1
        if (relocSuccess == false ) return -1;
        // the frame for merging should be keyframe
        if (cloudKeyPoses3D->empty() || goodToMergeMap == true)    return isIndoorJudgement();
        // allow overlapped area to display loop closures
        Eigen::Affine3f transStart;
        // if (localizationMode)
        // {
        //     if (temporaryCloudKeyPoses6D->empty() == true ) return isIndoorJudgement(); 
        //     transStart = pclPointToAffine3f(temporaryCloudKeyPoses6D->back());
        // }
        // else
            transStart = pclPointToAffine3f(cloudKeyPoses6D->back());

        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        float dist = sqrt(x*x + y*y + z*z);
        if (dist > surroundingkeyframeAddingDistThreshold) // for paranomic lidar angle threshold is not needed
            return isIndoorJudgement();
        // std::cout<<"dist: "<<dist<<endl;
        return -1;
    }

    int isIndoorJudgement()
    {
        // indoor means almost all points are confined
        int count = 0;
        int sizeP = lidarCloudSurfLastDS->points.size();
        for(int i = 0 ;i< sizeP; i++) 
        {
            PointType p = lidarCloudSurfLastDS->points[i];
            if (pointDistance(p) < 10.0) count++;
        }

        float ratio = count/(float)sizeP;
        int tmp = ratio> 0.2?1:0;
        // std::cout<<"ratio: " <<ratio<<" count: "<<count<<endl;
        return tmp;
    }


    void addOdomFactor()
    {
        if (cloudKeyPoses3D->empty())
        {
            // relax only z height!
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-3, 1e-3, 1e-3, 2e-2, 2e-2, 2e-2).finished()); 
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }
        else
        {

            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }


    void addGPSFactor()
    {
        if (gpsQueue.empty() || cloudKeyPoses3D->empty())
        {
            // std::cout<<gpsQueue.size()<<" "<<cloudKeyPoses3D->size()<<" "<<useGPS<<endl;
            return;
        }
            
        if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
        {
            // std::cout<<"key pose distance is within 5 meter"<<endl;
            return;
        }
            

        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
        {
            // std::cout<<"pose cov is still very small, no need for gps correction"<<endl;
            return;
        }
            
        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // use the last keypose, instead of the newest!
            // otherwise, very few gps factors can be added!
            int cloudSize = cloudKeyPoses6D->points.size();
            if(cloudSize < 2) break;
            if (gpsQueue.front().header.stamp.toSec() < cloudKeyPoses6D->points[cloudSize-1].time - 0.1)
            {
                // std::cout<<"the oldest message is too old, pop it out"<<endl;
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > cloudKeyPoses6D->points[cloudSize-1].time + 0.1)
            {
                // std::cout<<" all messages are too new" << endl;
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // if ( gps_variance[0] > gpsCovThreshold ||  gps_variance[1] > gpsCovThreshold )
                //     continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;

                if (abs(gps_x - cloudKeyPoses6D->points[cloudSize-1].x) >10.0 && abs(gps_y-cloudKeyPoses6D->points[cloudSize-1].y) > 10.0)
                {
                    // very rare
                    std::cout<<"too much difference between gps and lidar slam; x diff: "<<abs(gps_x - cloudKeyPoses6D->points[cloudSize-1].x)<<
                        "y diff: "<<abs(gps_y -cloudKeyPoses6D->points[cloudSize-1].y)<<endl;
                    continue;
                }
                    

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z ; // for body-lidar coor. trans.

                if(!useGpsElevation)
                {
                    curGPSPoint.z = cloudKeyPoses6D->points[cloudSize-1].z;
                }
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                // if(useGPS == false)
                // {
                //     Eigen::Affine3f gps_affine;
                //     odometryMsgToAffine3f(thisGPS, gps_affine);
                    
                //     gtsam::Vector var_vec_gtsam(6);
                //     var_vec_gtsam << gps_variance[0], gps_variance[1], gps_variance[2],gps_variance[3], gps_variance[4], gps_variance[5]; 
                //     noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances(var_vec_gtsam); 
                //     gtSAMgraph.add(PriorFactor<Pose3>(cloudSize-1, Affine3f2gtsamPose(gps_affine), priorNoise));
                    
                //     ROS_INFO("add 6 axis gps factor...");
                // }
                // else{
                    gtsam::Vector var_vec_gtsam(3);
                    var_vec_gtsam << gps_variance[3], gps_variance[4], gps_variance[5]; 
                    noiseModel::Diagonal::shared_ptr gps_variance = noiseModel::Diagonal::Variances(var_vec_gtsam);
                    gtsam::GPSFactor gps_factor(cloudSize-1, gtsam::Point3(gps_x, gps_y, gps_z), gps_variance);
                    gtSAMgraph.add(gps_factor);
                    aLoopIsClosed = true;
                    ROS_INFO("add 3 axis gps factor...");
                // }

                break;
            }
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            auto noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveTemporaryKeyframes()
    {
        //save temporary key poses
        int isIndoorJudgement = saveFrame();
        if ( isIndoorJudgement < 0 || frameTobeAbandoned) return;

        isIndoorKeyframeTMM.push_back(isIndoorJudgement);   
        int temporaryKeyPoseSize = temporaryCloudKeyPoses3D->size();
        PointType thisPose3D;
        PointTypePose thisPose6D;
        thisPose3D.x = transformTobeMapped[3];
        thisPose3D.y = transformTobeMapped[4];
        thisPose3D.z = transformTobeMapped[5];
        thisPose3D.intensity = temporaryKeyPoseSize; // this can be used as keyframe index

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = transformTobeMapped[0];
        thisPose6D.pitch = transformTobeMapped[1];
        thisPose6D.yaw   = transformTobeMapped[2];
        thisPose6D.time = cloudInfoTime;

        temporaryCloudKeyPoses3D->push_back(thisPose3D);
        temporaryCloudKeyPoses6D->push_back(thisPose6D);
        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*lidarCloudCornerLast,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*lidarCloudSurfLast,    *thisSurfKeyFrame);
        // save key frame cloud
        temporaryCornerCloudKeyFrames.push_back(thisCornerKeyFrame); // 这个全局都存着，但每次局部匹配只搜索50m内的关键帧
        temporarySurfCloudKeyFrames.push_back(thisSurfKeyFrame);

        if (!temporaryMappingMode && temporaryKeyPoseSize > slidingWindowSize) // sliding-window local map
        {
            temporaryCloudKeyPoses3D->erase(temporaryCloudKeyPoses3D->begin());
            temporaryCloudKeyPoses6D->erase(temporaryCloudKeyPoses6D->begin());
            temporaryCornerCloudKeyFrames.erase(temporaryCornerCloudKeyFrames.begin());
            temporarySurfCloudKeyFrames.erase(temporarySurfCloudKeyFrames.begin());
            isIndoorKeyframeTMM.erase(isIndoorKeyframeTMM.begin());
            // reindexing: key poses and key frames are corresponding with respect to the adding sequence
            for (int i = 0 ; i< (int)temporaryCloudKeyPoses3D->size(); i++)
            {
                temporaryCloudKeyPoses3D->points[i].intensity = i;
                temporaryCloudKeyPoses6D->points[i].intensity = i;
            }
        }

    }

    void saveKeyFramesAndFactor()
    {
        int indoorJudgement = saveFrame();
        if ( indoorJudgement < 0)
            return;
        
        isIndoorKeyframe.push_back(indoorJudgement);
        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // std::cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        Pose3 latestEstimate;
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // std::cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("gtsam current estimate: ");

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as keyframe index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = cloudInfoTime;
        cloudKeyPoses6D->push_back(thisPose6D);

        // std::cout << "****************************************************" << endl;
        // std::cout << "Pose covariance:" << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
        // std::cout<<"x cov: "<< poseCovariance(3,3)<< " y cov: "<<poseCovariance(4,4)<<endl;

        // std::cout<<"Before opt: "<<transformTobeMapped[3]<<endl;;
        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        affine_imu_to_map = trans2Affine3f(transformTobeMapped);
        // reset fusion
        // global_estimator->resetOptimization(affine_imu_to_map.matrix().cast<double>());

        // std::cout<<"After opt: "<< transformTobeMapped[3]<<endl;
        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());       
        // if pushing the original, it changes in the vec along with lidarCloudCornerLast
        pcl::copyPointCloud(*lidarCloudCornerLast,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*lidarCloudSurfLast,    *thisSurfKeyFrame); 
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame); 
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        updatePath(thisPose6D);

    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            lidarCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = mapFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();
        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry lidarOdometryROS;
        lidarOdometryROS.header.stamp = timeLidarInfoStamp;
        lidarOdometryROS.header.frame_id = mapFrame;
        lidarOdometryROS.child_frame_id = odometryFrame;
        lidarOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        lidarOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        lidarOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        // std::cout<<transformTobeMapped[3]<<" "<<transformTobeMapped[4]<<" "<<endl;
        lidarOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLidarOdometryGlobal.publish(lidarOdometryROS);
        // globalOdometry.push_back(lidarOdometryROS);
    }

    void publishLocalMap()
    {
        pcl::PointCloud<PointType>::Ptr cloudLocal(new pcl::PointCloud<PointType>());
        if (temporaryMappingMode == false)
        {        
            *cloudLocal += *lidarCloudSurfFromMap;
            *cloudLocal += *lidarCloudCornerFromMap; 
        }
        else
        {
            for (int i=0;i<(int)temporaryCloudKeyPoses3D->size();i++)
            {
                int idx = temporaryCloudKeyPoses3D->points[i].intensity;
                *cloudLocal += *transformPointCloud(temporarySurfCloudKeyFrames[idx],&temporaryCloudKeyPoses6D->points[i]);
                *cloudLocal += *transformPointCloud(temporaryCornerCloudKeyFrames[idx],&temporaryCloudKeyPoses6D->points[i]);
            }
        }
        publishCloud(&pubRecentKeyFrames, cloudLocal, timeLidarInfoStamp, mapFrame);

        //publish temporary keyposes for visualization
        publishCloud(&pubKeyPosesTmp,temporaryCloudKeyPoses3D,timeLidarInfoStamp, mapFrame);

        globalPath.header.stamp = timeLidarInfoStamp;
        globalPath.header.frame_id = mapFrame;
        pubPath.publish(globalPath);
    }

    void getCPUinfoThread()
    {
        ros::Rate rate(0.5);
        while(ros::ok())
        {
            current_pid = GetCurrentPid(); // or you can set a outside program pid
            cpu_usage_ratio = GetCpuUsageRatio(current_pid);
            memory_usage = GetMemoryUsage(current_pid);

            // if(debugMode) std::cout << "current pid: " << current_pid << std::endl;
            // if(debugMode) std::cout << "cpu usage ratio: " << cpu_usage_ratio * 100 << "%" << std::endl;
            // if(debugMode) std::cout << "memory usage: " << memory_usage << "MB" << std::endl;
            rate.sleep();
        }
        
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roll_semi");

    mapOptimization MO_semi;

    ROS_INFO("\033[1;32m----> Map Optimization for semi-auto Started.\033[0m");
    
    // std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    // std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);
    std::thread mappingThread{&mapOptimization::run,&MO_semi};
    std::thread getCPUinfoThread(&mapOptimization::getCPUinfoThread, &MO_semi);

    ros::spin();

    // loopthread.join();
    // visualizeMapThread.join();
    mappingThread.join();
    getCPUinfoThread.join();
    
    return 0;
}