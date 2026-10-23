#include "utility.h"
#include "roll/cloud_info.h"
#include "roll/save_map.h"
#include "getCpuOccupancy.h"
#include "gnssLiDARinitializer.h"

#include"LOAMmapping.h"
#include "globalOpt.h"
#include "MapStorage.h"
#include "FixedMapSubmap.h"
#include "LocalizationGate.h"
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <social_nav_localization/LocalizationState.h>
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
    float cpu_usage_ratio = 0;
    float memory_usage = 0;

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
    Eigen::Affine3f rvizInitialPose;

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

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subGT;
    ros::Subscriber subLoop;
    ros::Subscriber subLidarOdometry;
    ros::Subscriber initialpose_sub;
    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    
    roll::cloud_info cloudInfo;
    struct PendingCloud {
        roll::cloud_infoConstPtr message;
        std::chrono::steady_clock::time_point received;
    };
    queue<PendingCloud> cloudInfoBuffer;
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
    map<int, uint64_t> keyframeCacheAccess;
    size_t keyframeCacheBytes = 0;
    uint64_t cacheSequence = 0;
    bool globalDisplayCached = false;
    vector<int> cachedLocalKeyframes;
    Eigen::Vector3f localSubmapCenter = Eigen::Vector3f::Zero();
    pcl::KdTreeFLANN<PointType>::Ptr localCornerTree, localSurfaceTree;
    std::unique_ptr<LOAMmapping> matcher;
    std::unique_ptr<FixedMapSubmap<PointTypePose>> fixedSubmap;
    uint64_t submapEpoch = 0;
    bool submapPending = false;
    double lastSubmapBuildMs = 0;
    uint64_t submapBuilds = 0, cacheMisses = 0;
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
    std::mutex queueMutex;
    ros::ServiceServer srvStatus;
    bool processing = false;
    uint64_t receivedClouds = 0, receivedOdometry = 0, processedFrames = 0;
    uint64_t droppedClouds = 0, droppedOdometry = 0;
    uint64_t lastProcessedStamp = 0;
    std::condition_variable queueReady;
    std::recursive_mutex mapDataMutex;
    std::mutex saveMutex;
    size_t keyposeTreeSize = 0;
    double lastInlierRatio = 0;
    double lastRegistrationError = -1;
    bool matchingValid = false;
    LocalizationGate localizationGate;
    Eigen::Affine3f gatePrediction = Eigen::Affine3f::Identity();
    ros::Publisher pubTrackingStatus;
    ros::Publisher pubInitialPoseAccepted;
    bool waitForInitialPose = false, initialPoseReceived = false;
    ros::Subscriber subFrontendReference;
    bool frontendInitialization = false, frontendReferenceReady = false, frontendReferenceApplied = false;
    Eigen::Affine3f frontendReference = Eigen::Affine3f::Identity();
    ros::Publisher pubLocalizationState;
    std::string localFrame, localizationSession;
    bool stationaryImuAssumption = false;
    uint64_t localizationGeneration = 0, mapRevision = 0;
    bool publishLegacyOutputs = false, publishLocalizationTF = true;
    ofstream metricsFile;
    ofstream trajectoryFile;
    bool writeDetails = false;
    std::ofstream registrationFile;
    std::mutex mtxInit;
    std::mutex mtxLoopInfo;
    std::mutex pose_estimator_mutex;
    std::mutex mtxGPSqueue;


    int lidarCloudCornerFromMapDSNum = 0;
    int lidarCloudSurfFromMapDSNum = 0;
    int lidarCloudCornerLastDSNum = 0;
    int lidarCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    int lastLoopTraceKey = -1;
    
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

        gpsTrajPub      = nh.advertise<nav_msgs::Odometry> ("/roll/gps_odom", 1);

        pubPath                     = nh.advertise<nav_msgs::Path>("/roll/mapping/path", 1);
        pubPathFusion               = nh.advertise<nav_msgs::Path>("/roll/mapping/path_fusion", 1);
        pubPathFusionVINS               = nh.advertise<nav_msgs::Path>("/roll/mapping/path_fusion_vins", 1);
        pubLidarOdometryGlobalFusionBaselink      = nh.advertise<nav_msgs::Odometry> ("/roll/mapping/odometry_fusion_baselink", 1);
        pubPathGPS                     = nh.advertise<nav_msgs::Path>("/roll/mapping/gps", 1);

        initialpose_sub = nh.subscribe("/initialpose", 1, &mapOptimization::initialpose_callback, this);

        pubKeyPosesTmp                 = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/tmp_key_poses", 1);
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/key_poses", 1);
        pubLidarCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/map_global", 1, true);
        pubLidarOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("/roll/mapping/odometry", 1);
        pubLidarOdometryGlobalFusion      = nh.advertise<nav_msgs::Odometry> ("/roll/mapping/odometry_fusion", 1);
        vinsFusion      = nh.advertise<nav_msgs::Odometry> ("/roll/mapping/vins_fusion", 1);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/roll/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/map_local", 1);

        pubMergedMap = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/merged_map", 1);

        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("/roll/mapping/cloud_registered", 1);

        subCloud = nh.subscribe<roll::cloud_info>("/roll/feature/cloud_info", 64, &mapOptimization::lidarCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = nh.subscribe<sensor_msgs::NavSatFix> (gpsTopic, 200, &mapOptimization::gpsHandler, this);
        subGT   = nh.subscribe<nav_msgs::Odometry> (gtTopic, 200, &mapOptimization::gtHandler, this); 
        subLidarOdometry = nh.subscribe<nav_msgs::Odometry> (odometryTopic, 64, &mapOptimization::lidarOdometryHandler,this);

        srvSaveMap  = nh.advertiseService("/roll/save_map", &mapOptimization::saveMapService, this);
        srvStatus = nh.advertiseService("/roll/status", &mapOptimization::statusService, this);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization


        allocateMemory();
        pubTrackingStatus = nh.advertise<std_msgs::String>("/roll/tracking_status", 1, true);
        std::string stateTopic;
        nh.param<std::string>("roll/localizationStateTopic", stateTopic, "/localization/state");
        nh.param<std::string>("roll/localFrame", localFrame, "odom");
        nh.param<bool>("stationary_imu/enabled", stationaryImuAssumption, false);
        nh.param<bool>("roll/publishLegacyOutputs", publishLegacyOutputs, false);
        nh.param<bool>("roll/publishLocalizationTF", publishLocalizationTF, true);
        nh.param<bool>("roll/writeDetails", writeDetails, false);
        nh.param<bool>("roll/waitForInitialPose", waitForInitialPose, false);
        nh.param<bool>("roll/coverageGateEnabled", localizationGate.enabled, false);
        localizationGate.enabled = localizationGate.enabled && localizationMode;
        if (localizationGate.enabled) {
            XmlRpc::XmlRpcValue polygons;
            std::string coverageFrame;
            nh.param<std::string>("roll/localization_coverage/frame_id", coverageFrame, "");
            if (coverageFrame != mapFrame || !nh.getParam("roll/localization_coverage/regions_xy", polygons) ||
                    polygons.getType() != XmlRpc::XmlRpcValue::TypeArray || polygons.size() == 0)
                throw std::runtime_error("Missing map-frame localization coverage polygons");
            auto number = [](XmlRpc::XmlRpcValue& value) -> double {
                if (value.getType() == XmlRpc::XmlRpcValue::TypeInt) return static_cast<int>(value);
                if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble) return static_cast<double>(value);
                throw std::runtime_error("Non-numeric coverage coordinate");
            };
            for (int i = 0; i < polygons.size(); ++i) {
                auto& poly = polygons[i];
                if (poly.getType() != XmlRpc::XmlRpcValue::TypeArray || poly.size() < 3)
                    throw std::runtime_error("Invalid coverage polygon");
                std::vector<Eigen::Vector2f> vertices;
                for (int j = 0; j < poly.size(); ++j) {
                    if (poly[j].getType() != XmlRpc::XmlRpcValue::TypeArray || poly[j].size() != 2)
                        throw std::runtime_error("Invalid coverage vertex");
                    Eigen::Vector2f vertex(number(poly[j][0]), number(poly[j][1]));
                    if (!vertex.allFinite()) throw std::runtime_error("Nonfinite coverage vertex");
                    vertices.push_back(vertex);
                }
                localizationGate.regions.push_back(vertices);
            }
            nh.param("roll/coverageGate/reentryMargin", localizationGate.reentryMargin, .5);
            nh.param("roll/coverageGate/minInlier", localizationGate.minInlier, .55);
            nh.param("roll/coverageGate/maxResidual", localizationGate.maxResidual, .10);
            nh.param("roll/coverageGate/maxStep", localizationGate.maxStep, .75);
            nh.param("roll/coverageGate/maxAngle", localizationGate.maxAngle, .15);
            nh.param("roll/coverageGate/maxGravity", localizationGate.maxGravity, .0872665);
            nh.param("roll/coverageGate/consistencyDistance", localizationGate.consistencyDistance, .25);
            nh.param("roll/coverageGate/consistencyAngle", localizationGate.consistencyAngle, .035);
            nh.param("roll/coverageGate/confirmationMatches", localizationGate.confirmationMatches, 5);
            if (temporaryMappingEnabled || mapUpdateEnabled)
                throw std::runtime_error("Coverage gating requires an immutable prior map");
        }
        nh.param<bool>("initialization/enabled", frontendInitialization, false);
        subFrontendReference = nh.subscribe<geometry_msgs::PoseStamped>("/fast_lio/initial_lidar_pose", 1,
            [this](const geometry_msgs::PoseStamped::ConstPtr& msg) {
                const auto& p=msg->pose.position; const auto& q=msg->pose.orientation;
                Eigen::Quaternionf rotation(q.w,q.x,q.y,q.z);
                if (msg->header.frame_id!="camera_init" || !rotation.coeffs().allFinite() ||
                    std::abs(rotation.norm()-1)>1e-3 || !std::isfinite(p.x+p.y+p.z)) return;
                std::lock_guard<std::mutex> lock(queueMutex);
                if (frontendReferenceReady) return;
                frontendReference.linear()=rotation.toRotationMatrix();
                frontendReference.translation()=Eigen::Vector3f(p.x,p.y,p.z);
                frontendReferenceReady=true;
                queueReady.notify_one();
            });
        if (localizationMode)
            pubInitialPoseAccepted = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/roll/initial_pose_accepted", 8, true);
        if (localFrame.empty() || localFrame == mapFrame || localFrame == lidarFrame)
            throw std::runtime_error("Continuous, map and LiDAR frames must be distinct");
        localizationSession = std::to_string(ros::WallTime::now().toNSec());
        pubLocalizationState = nh.advertise<social_nav_localization::LocalizationState>(stateTopic, 8);
        std::filesystem::create_directories(log_dir);
        if (saveLog) pose_log_file.open(log_dir + "/matching.log");
        if (useGPS) gps_file.open(log_dir + "/gps.txt");
        if (writeDetails) {
            registrationFile.open(log_dir + "/registration.csv");
            registrationFile << "stamp_ns,iteration,correspondences,degenerate,min_eigen,residual,"
                                "raw_rotation_rad,raw_translation_m,guess_roll,guess_pitch,guess_yaw,"
                                "guess_x,guess_y,guess_z,step_roll,step_pitch,step_yaw,step_x,step_y,step_z,"
                                "eigen_0,eigen_1,eigen_2,eigen_3,eigen_4,eigen_5,weak_roll,weak_pitch,"
                                "weak_yaw,weak_x,weak_y,weak_z\n";
            metricsFile.open(log_dir + "/timings.csv");
            metricsFile << "stamp,processing_ms,corners,surfaces,inlier_ratio,registration_error,valid,"
                           "submap_ms,matching_ms,local_keyframes,map_points,submap_builds,cache_misses,"
                           "stamp_ns,queue_wait_ms,map_lock_ms,decode_ms,publish_ms,submap_build_ms,"
                           "gate_reason,predicted_x,predicted_y,predicted_z\n";
            trajectoryFile.open(log_dir + "/trajectory.tum");
            trajectoryFile << std::setprecision(12);
        }
        if ((saveLog && !pose_log_file) || (writeDetails && (!metricsFile || !trajectoryFile || !registrationFile)))
            throw std::runtime_error("Cannot open run outputs: " + log_dir);

        namespace fs = std::filesystem;
        if (!localizationMode) {
            fs::path mapRoot = fs::path(saveKeyframeMapDirectory).parent_path();
            if (fs::exists(mapRoot) && !fs::is_empty(mapRoot))
                throw std::runtime_error("Refusing existing non-empty map directory: " + mapRoot.string());
            fs::create_directories(saveKeyframeMapDirectory);
        } else {
            roll_storage::load(saveKeyframeMapDirectory, *cloudKeyPoses6D, *cloudKeyPoses3D,
                               cornerCloudKeyFrames, surfCloudKeyFrames, isIndoorKeyframe);
            mapLoaded = true;
            ROS_INFO("Loaded %zu keyframes from %s", cloudKeyPoses6D->size(), saveKeyframeMapDirectory.c_str());
            bool asynchronous;
            nh.param<bool>("roll/localizationAsyncSubmap", asynchronous, true);
            if (asynchronous && useKeyframeCache && localizationSubmapUpdateDistance > 0 &&
                !mapUpdateEnabled && !temporaryMappingEnabled) {
                fixedSubmap.reset(new FixedMapSubmap<PointTypePose>(*cloudKeyPoses6D, cornerCloudKeyFrames,
                    surfCloudKeyFrames, surroundingKeyframeSearchRadius, surroundingKeyframeDensity,
                    mappingCornerLeafSize, mappingSurfLeafSize, keyframeCacheMaxEntries, keyframeCacheMaxBytes));
                if (!waitForInitialPose)
                    prepareFixedSubmap(Eigen::Vector3f(initialGuess[3], initialGuess[4], initialGuess[5]));
            }
        }

    }

    void allocateMemory()
    {        
        
        
        resetISAM();
        affine_imu_to_body = trans2Affine3f(array_imu_to_body);
        affine_lidar_to_body =  trans2Affine3f(array_lidar_to_body);
        affine_gps_to_body =  trans2Affine3f(array_gps_to_body);

        affine_imu_to_lidar = affine_lidar_to_body.inverse()*affine_imu_to_body;
        
        affine_lidar_to_gps = affine_gps_to_body.inverse()*affine_lidar_to_body;
        cout<<"affine_lidar_to_gps: "<<affine_lidar_to_gps.matrix()<<endl;

        global_estimator = std::make_shared<GlobalOptimization>(slidingWindowSizeFusion, enableSecondaryFusion);
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

    bool statusService(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& response)
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        response.success = droppedClouds == 0 && droppedOdometry == 0;
        std::ostringstream out;
        out << "{\"busy\":" << (processing ? "true" : "false")
            << ",\"pending_clouds\":" << cloudInfoBuffer.size()
            << ",\"pending_odometry\":" << lidarOdometryBuffer.size()
            << ",\"received_clouds\":" << receivedClouds
            << ",\"received_odometry\":" << receivedOdometry
            << ",\"processed\":" << processedFrames
            << ",\"last_processed_stamp_ns\":" << lastProcessedStamp
            << ",\"dropped_clouds\":" << droppedClouds
            << ",\"dropped_odometry\":" << droppedOdometry << '}';
        response.message = out.str();
        return true;
    }

    void lidarCloudInfoHandler(const roll::cloud_infoConstPtr& msgIn)
    {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            cloudInfoBuffer.push({msgIn, std::chrono::steady_clock::now()});
            ++receivedClouds;
            if (cloudInfoBuffer.size() > 128) {
                ROS_ERROR_THROTTLE(1, "ROLL input queue overflow: dropping oldest cloud");
                cloudInfoBuffer.pop();
                ++droppedClouds;
            }
        }
        queueReady.notify_one();
    }

    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& msgIn)
    {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            lidarOdometryBuffer.push(msgIn);
            ++receivedOdometry;
            if (lidarOdometryBuffer.size() > 128) {
                ROS_ERROR_THROTTLE(1, "ROLL input queue overflow: dropping oldest odometry");
                lidarOdometryBuffer.pop();
                ++droppedOdometry;
            }
        }
        queueReady.notify_one();
        std::lock_guard<std::recursive_mutex> mapLock(mapDataMutex);
        
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
        if (publishLegacyOutputs) br.sendTransform(trans_odom_to_lidar);

        tf::Transform t_odom_to_lidar_foot = tf::Transform(tf::createQuaternionFromRPY(0,0, array_imu_to_map[2]),tf::Vector3(array_imu_to_map[3],
            array_imu_to_map[4], 0.2));
        tf::StampedTransform trans_odom_to_lidar_foot = tf::StampedTransform(t_odom_to_lidar_foot, msgIn->header.stamp, mapFrame, "lidar_footprint");
        if (publishLegacyOutputs) br.sendTransform(trans_odom_to_lidar_foot);
        
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
        if (visualizationPathLimit > 0 && globalPathFusionVINS.poses.size() > static_cast<size_t>(visualizationPathLimit))
            globalPathFusionVINS.poses.erase(globalPathFusionVINS.poses.begin());
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
        // cout<<"odom to baselink rpy XYZ: "<<r<<" "<<p<<" "<<y<<" "<<t_odom_to_baselink.getOrigin()[0]<<" "<<t_odom_to_baselink.getOrigin()[1]
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
        if (publishLegacyOutputs) pubLidarOdometryGlobalFusionBaselink.publish(odomAftMappedBaselink);
        

        float odomTmp[6];
        Affine3f2Trans(affine_lidar_to_lidar0, odomTmp);
        // cout<<"odom message: "<<odomTmp[3]<<" "<< odomTmp[4]<<endl;
        
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
        if (visualizationPathLimit > 0 && globalPathFusion.poses.size() > static_cast<size_t>(visualizationPathLimit))
            globalPathFusion.poses.erase(globalPathFusion.poses.begin());
        globalPathFusion.header.stamp = msgIn->header.stamp;
        globalPathFusion.header.frame_id = mapFrame;
        if(pubPathFusion.getNumSubscribers() != 0) 
            pubPathFusion.publish(globalPathFusion); // before loop closure
    }

    void transformUpdate()
    {
        mtx.lock();
        affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse();

        // cout<<"affine_imu_to_map "<<affine_imu_to_map.matrix()<<endl;
        // cout<<"affine_odom_to_map "<<affine_odom_to_map.matrix()<<endl;
        // // Publish TF
        // static tf::TransformBroadcaster br;
        // tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
        //                                               tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        // // child frame 'lidar_link' expressed in parent_frame 'mapFrame'
        // tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLidarInfoStamp, mapFrame, lidarFrame);
        // br.sendTransform(trans_odom_to_lidar);
        mtx.unlock();
    }
    void run()
    {
        ros::Rate matchingRate(globalMatchingRate);
        while(ros::ok()){ // why while(1) is not okay???
            {
                roll::cloud_infoConstPtr cloudInfoMsg;
                nav_msgs::Odometry::ConstPtr lidarOdometryMsg;
                std::chrono::steady_clock::time_point received;
                {
                    std::unique_lock<std::mutex> queueLock(queueMutex);
                    queueReady.wait_for(queueLock, std::chrono::milliseconds(100), [this] {
                        return (!frontendInitialization || frontendReferenceReady) &&
                            !cloudInfoBuffer.empty() && !lidarOdometryBuffer.empty();
                    });
                    if (!ros::ok()) break;
                    if (cloudInfoBuffer.empty() || lidarOdometryBuffer.empty()) continue;
                    if (frontendInitialization && !frontendReferenceReady) continue;
                    double cloudTime = cloudInfoBuffer.front().message->header.stamp.toSec();
                    double odomTime = lidarOdometryBuffer.front()->header.stamp.toSec();
                    if (std::abs(cloudTime - odomTime) > 0.005) {
                        if (cloudTime < odomTime) cloudInfoBuffer.pop(); else lidarOdometryBuffer.pop();
                        continue;
                    }
                    cloudInfoMsg = cloudInfoBuffer.front().message;
                    received = cloudInfoBuffer.front().received; cloudInfoBuffer.pop();
                    lidarOdometryMsg = lidarOdometryBuffer.front(); lidarOdometryBuffer.pop();
                    if (latestScanOnly) {
                        while (cloudInfoBuffer.size() > 1) cloudInfoBuffer.pop();
                    }
                    processing = true;
                }
                const auto dequeued = std::chrono::steady_clock::now();
                std::lock_guard<std::recursive_mutex> mapLock(mapDataMutex);
                const auto locked = std::chrono::steady_clock::now();
                const double queueMs = std::chrono::duration<double, std::milli>(dequeued-received).count();
                const double lockMs = std::chrono::duration<double, std::milli>(locked-dequeued).count();
                TicToc decode;
                if (localizationMode && waitForInitialPose && !initialPoseReceived && !poseGuessFromRvizAvailable) {
                    std::lock_guard<std::mutex> queueLock(queueMutex);
                    processing = false;
                    continue; // Never initialize from the placeholder while waiting for selection.
                }
                timeLidarInfoStamp = cloudInfoMsg->header.stamp;
                cloudInfoTime = timeLidarInfoStamp.toSec();
                if (rosTimeStart < 0) rosTimeStart = cloudInfoTime;
                cloudInfo = *cloudInfoMsg;
                Eigen::Affine3f tmp;
                odometryMsgToAffine3f(*lidarOdometryMsg, tmp);
                affine_imu_to_odom = affine_imu_to_lidar * tmp;
                pcl::fromROSMsg(cloudInfoMsg->cloud_corner, *lidarCloudCornerLast);
                pcl::fromROSMsg(cloudInfoMsg->cloud_surface, *lidarCloudSurfLast);
                pcl::fromROSMsg(cloudInfoMsg->cloud_raw, *lidarCloudRaw);
                const double decodeMs = decode.toc();
                TicToc mapping;
                double submapMs = 0, matchingMs = 0;

                matchingValid = false;
                lastInlierRatio = 0;
                lastRegistrationError = -1;
                updateInitialGuess(); // preserve original local initialization and tracking
                gatePrediction = affine_imu_to_map;
                const bool insideCoverage = localizationGate.begin(gatePrediction);

                if (insideCoverage && (tryReloc == true || relocSuccess == true))
                {
                    
                    TicToc extract;
                    extractNearby();
                    submapMs = extract.toc();
                    
                    if(debugMode) cout<<"extract: "<<extract.toc()<<endl;
                    TicToc downsample;
                    downsampleCurrentScan();

                    if(debugMode) cout<<"downsample: "<<downsample.toc()<<endl;
                    TicToc opt;
                    
                    scan2MapOptimization();
                    
                    float optTime = opt.toc();
                    matchingMs = optTime;
                    if (localizationMode && !matchingValid) cachedLocalKeyframes.clear();
                    if(debugMode)  cout<<"optimization: "<<optTime<<endl; // > 90% of the total time
                    
                    TicToc optPose;
                    if (localizationMode)
                    {
                        if (temporaryMappingEnabled) saveTemporaryKeyframes();
                        updatePathRELOC(cloudInfoMsg);  // for visualizing in rviz
                    }
                    else
                    {
                        saveKeyFramesAndFactor();
                        correctPoses();
                    }
                    float optPoseTime = optPose.toc();
                    if(debugMode)  cout<<"pose opt. takes "<< optPoseTime<<endl;
                

                }
                
                TicToc publish;
                publishLocalMap();
                publishOdometry();
                transformUpdate();
                publishLocalizationState(*lidarOdometryMsg);
                const double publishMs = publish.toc();
                
                frameTobeAbandoned = false;
                if(debugMode) printTrans("after mapping: ",transformTobeMapped);
                // cout<<"publish: "<<publish.toc()<<endl;
                // cout<<"after mapping z diff: "<<transformTobeMapped[2]-transformBeforeMapped[2]<<endl;


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
                    // cout<<temporaryCloudKeyPoses3D->size()<<endl;
                    // cout<<"reindexing: key poses and key frames are corresponding with respect to the adding sequence"<<endl;
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
                if (writeDetails) metricsFile << std::setprecision(12) << cloudInfoTime << ',' << mappingTimeCur << ','
                            << lidarCloudCornerLastDSNum << ',' << lidarCloudSurfLastDSNum << ','
                            << lastInlierRatio << ',' << lastRegistrationError << ',' << matchingValid << ','
                            << submapMs << ',' << matchingMs << ',' << cachedLocalKeyframes.size() << ','
                            << lidarCloudCornerFromMapDSNum + lidarCloudSurfFromMapDSNum << ','
                            << submapBuilds << ',' << cacheMisses << ',' << timeLidarInfoStamp.toNSec() << ','
                            << queueMs << ',' << lockMs << ',' << decodeMs << ',' << publishMs << ','
                            << lastSubmapBuildMs << ',' << localizationGate.reason << ','
                            << gatePrediction.translation().x() << ',' << gatePrediction.translation().y() << ','
                            << gatePrediction.translation().z() << '\n';
                {
                    std::lock_guard<std::mutex> queueLock(queueMutex);
                    ++processedFrames;
                    lastProcessedStamp = timeLidarInfoStamp.toNSec();
                    processing = false;
                }
                std_msgs::String status;
                status.data = localizationMode ? (matchingValid ? "TRACKING" : "LOST") : "MAPPING";
                pubTrackingStatus.publish(status);
                if(debugMode)  cout<<"mapping time: "<<mappingTimeCur<<endl;

                if(globalMatchingRate <= 4.0) // otherwise it is just waste of time
                    matchingRate.sleep();
            }


        }

    }

    void mergeMap()
    {
        cout<<" DO gtsam optimization here"<<endl;
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

        // cout<<" add prior factor instead to constrain the covariances"<<endl;
        // cannot put it in the loop above
        // odomFactor needs to be a smooth one!!!
        
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[tempSize - 2 ]);
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[tempSize - 1 ]);
        gtSAMgraphTM.add(BetweenFactor<Pose3>(tempSize - 2 , tempSize -1, poseFrom.between(poseTo), odometryNoise));

        
        gtSAMgraphTM.add(PriorFactor<Pose3>(tempSize - 1, poseCorr, corrNoise));
        initialEstimateTM.insert(tempSize - 1, poseCorr);
        
        // cout<<"before opt. "<< poseCorr.translation().x()<<" "<< poseCorr.translation().y()<<" "<< poseCorr.translation().z()<<endl;

        // update iSAM
        isamTM->update(gtSAMgraphTM, initialEstimateTM);
        isamTM->update();
        isamTM->update();
        isamTM->update();
        gtSAMgraphTM.resize(0);
        initialEstimateTM.clear();

        
        isamCurrentEstimateTM = isamTM->calculateEstimate();

        // cout<<"pose correction: "<<isamCurrentEstimateTM.size()<<endl;

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
            // cout<<"nearest: "<<keyPoseSearchDist.size()<<endl;
            mtx.lock();
            if (keyPoseSearchDist[0] < 2*surroundingKeyframeDensity)
            {
                // cout<<keyPoseSearchIdx[0]<<endl;
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
        cout<<"map merge takes "<<t_merge.toc()<< " ms"<<endl; // negligible

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
            if(debugMode) cout<<"Got 6axis gnss message or bad status: "<<gpsMsg->status.status<<endl;
            return;
        }

        double x,y,z; // ENU, note that in nclt, it it NED
        x = sin(deg2rad(gpsMsg->longitude - longi0))*rew*cos(deg2rad(lati0));
        y = sin(deg2rad(gpsMsg->latitude - lati0))*rns;
        z = gpsMsg->altitude -alti0;

        // gps receiver is highly overlapped with lidar center in xy, so neglecting their extrensics
        Eigen::Affine3f affine_gps_to_enu = pcl::getTransformation(x,y,z,0,0,0);

        Eigen::Affine3f affine_lidar_to_enu = affine_gps_to_enu*affine_lidar_to_gps;
        
        if(debugMode) cout<<"after conversion x y z: "<<affine_lidar_to_enu(0,3)<<" "<<affine_lidar_to_enu(1,3)<<" "<<affine_lidar_to_enu(2,3)<<endl;
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
            if(debugMode) cout<<"x: "<< gtMsg->pose.pose.position.x<<" cov: "<<gtMsg->pose.covariance[0]<<endl;
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
        res.success = false;
        if (localizationMode) {
            ROS_ERROR("Cannot save/overwrite a loaded localization map");
            return true;
        }
        std::lock_guard<std::mutex> saving(saveMutex);
        if (doneSavingMap) {
            ROS_WARN("This session's map is already saved; start a new session for another map snapshot");
            return true;
        }
        try {
            pcl::PointCloud<PointTypePose> poses;
            vector<pcl::PointCloud<PointType>::Ptr> corners, surfaces;
            vector<int> indoor;
            {
                std::lock_guard<std::recursive_mutex> mapLock(mapDataMutex);
                poses = *cloudKeyPoses6D;
                indoor = isIndoorKeyframe;
                for (const auto& cloud : cornerCloudKeyFrames) corners.push_back(cloud->makeShared());
                for (const auto& cloud : surfCloudKeyFrames) surfaces.push_back(cloud->makeShared());
            }
            if (req.resPoseIndoor != 0 || req.resPoseOutdoor != 0 || req.overlapThre != 0)
                throw std::runtime_error("Use zero keyframe sparsification fields; complete keyframes are preserved");
            roll_storage::save(std::filesystem::path(saveKeyframeMapDirectory).parent_path(), poses,
                               corners, surfaces, indoor, req.resolutionMap == 0 ? 0.1 : req.resolutionMap, mapFrame);
            doneSavingMap = true;
            res.success = true;
            ROS_INFO("Saved complete map with %zu keyframes", poses.size());
        } catch (const std::exception& error) {
            ROS_ERROR("Map save failed: %s", error.what());
        }
        return true;
    }

    void keyframeSparsification(pcl::PointCloud<PointType>::Ptr  &cloudKeyPoses3DDS, float resMap, float resPoseIndoor, float resPoseOutdoor,
                                float overlapThre)
    {
        if(resPoseIndoor == 0 || resPoseOutdoor == 0 || overlapThre == 0) 
        {
            pcl::copyPointCloud(*cloudKeyPoses3D,*cloudKeyPoses3DDS);
            cout << "No downsampling performed on kefyrame map saving" << endl;
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
        cout<<"indoor: "<<keyPosesIndoorDS->size()<<" frames" <<endl;
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
         cout<<"outdoor: "<<keyPosesOutdoorDS->size()<<" frames" <<endl;
    }

    void visualizeGlobalMapThread()
    {
        ros::WallRate rateMap(0.2); // New viewers must receive the map even when bag /clock is paused.
        while (ros::ok())
        {
            {
                std::lock_guard<std::recursive_mutex> mapLock(mapDataMutex);
                if (pubKeyPoses.getNumSubscribers()) publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLidarInfoStamp, mapFrame);
            }           
            publishGlobalMap();
            rateMap.sleep();
        }

    }

    void publishGlobalMap()
    {
        pcl::PointCloud<PointType>::Ptr positions(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointTypePose>::Ptr poses(new pcl::PointCloud<PointTypePose>());
        vector<pcl::PointCloud<PointType>::Ptr> corners, surfaces;
        ros::Time stamp;
        uint64_t revision, generation;
        PointType pt;
        {
            std::lock_guard<std::recursive_mutex> mapLock(mapDataMutex);
            if (pubLidarCloudSurround.getNumSubscribers() == 0 || cloudKeyPoses3D->empty() ||
                (localizationMode && globalDisplayCached)) return;
            *positions = *cloudKeyPoses3D; *poses = *cloudKeyPoses6D;
            // Saved keyframe clouds are immutable; retain ownership across append/erase.
            corners = cornerCloudKeyFrames; surfaces = surfCloudKeyFrames;
            stamp = timeLidarInfoStamp; revision = mapRevision; generation = localizationGeneration;
            pt.x = affine_imu_to_map(0,3); pt.y = affine_imu_to_map(1,3); pt.z = affine_imu_to_map(2,3);
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
        kdtreeGlobalMap->setInputCloud(positions);
        kdtreeGlobalMap->radiusSearch(pt, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(positions->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key poses 
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        // fix the keyframe downsample bug
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = positions->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        // only for visualization
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
        {
            if (pointDistance(globalMapKeyPosesDS->points[i], positions->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(corners[thisKeyInd], &poses->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfaces[thisKeyInd], &poses->points[thisKeyInd]);
        }
        // downsample visualized points: why it is not working
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        // cout<<globalMapKeyFramesDS->size()<<endl;       
        sensor_msgs::PointCloud2 display;
        pcl::toROSMsg(*globalMapKeyFramesDS, display);
        display.header.stamp = stamp; display.header.frame_id = mapFrame;
        std::lock_guard<std::recursive_mutex> mapLock(mapDataMutex);
        if (revision != mapRevision || generation != localizationGeneration) return;
        pubLidarCloudSurround.publish(display);
        if (localizationMode) {
            globalDisplayCached = true;
        }

    }

    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false )
            return;

        ros::Rate rate(loopClosureFrequency);
        int nextKeyframe = 0;
        while (ros::ok())
        {
            rate.sleep();
            if (!localizationMode) 
            {
                // Polling only the latest pose can skip a short overlap window.
                // Consume all unchecked keyframes in order at each worker wake.
                while (ros::ok() && performLoopClosure(nextKeyframe))
                    ++nextKeyframe;
                visualizeLoopClosure();
            }
            
        }
    }


    bool performLoopClosure(int loopKeyCur)
    {
        std::unique_lock<std::recursive_mutex> mapLock(mapDataMutex);
        if (loopKeyCur >= static_cast<int>(cloudKeyPoses3D->size()))
            return false; // Wait for a new keyframe rather than retrying an unchanged scan.
        const uint64_t revision = mapRevision;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyPre;
        if (detectLoopClosureDistance(loopKeyCur, &loopKeyPre) == false) {
            if (writeDetails && lastLoopTraceKey != loopKeyCur)
                ROS_INFO_STREAM("loop_check stage=no_candidate current=" << loopKeyCur
                    << " t=" << copy_cloudKeyPoses6D->points[loopKeyCur].time-rosTimeStart << " revision=" << revision);
            lastLoopTraceKey = loopKeyCur;
            return true;
        }
        lastLoopTraceKey = loopKeyCur;
        const double checkTime = copy_cloudKeyPoses6D->points[loopKeyCur].time-rosTimeStart;
        if (writeDetails)
            ROS_INFO_STREAM("loop_check stage=candidate current=" << loopKeyCur << " previous=" << loopKeyPre
                << " t=" << checkTime << " distance=" << pointDistance(copy_cloudKeyPoses3D->points[loopKeyCur],
                    copy_cloudKeyPoses3D->points[loopKeyPre]) << " dt="
                << copy_cloudKeyPoses6D->points[loopKeyCur].time-copy_cloudKeyPoses6D->points[loopKeyPre].time
                << " revision=" << revision);

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0, loopKeyCur);
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, loopKeyCur);
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000) {
            if (writeDetails)
                ROS_INFO_STREAM("loop_check stage=insufficient_points current=" << loopKeyCur
                    << " previous=" << loopKeyPre << " source_points=" << cureKeyframeCloud->size()
                    << " target_points=" << prevKeyframeCloud->size());
            return true;
        }
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLidarInfoStamp, mapFrame);
        // ICP only accesses the already assembled snapshot clouds.
        mapLock.unlock();

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
        TicToc icpTimer;
        registration.align(*unused_result);

        const double score = registration.getFitnessScore();
        if (writeDetails) {
            const Eigen::Matrix4f correction = registration.getFinalTransformation();
            ROS_INFO_STREAM("loop_check stage=icp current=" << loopKeyCur << " previous=" << loopKeyPre
                << " t=" << checkTime << " source_points=" << cureKeyframeCloud->size()
                << " target_points=" << prevKeyframeCloud->size() << " converged=" << registration.hasConverged()
                << " score=" << score << " limit=" << historyKeyframeFitnessScore
                << " dx=" << correction(0,3) << " dy=" << correction(1,3) << " dz=" << correction(2,3)
                << " angle_rad=" << Eigen::AngleAxisf(correction.block<3,3>(0,0)).angle()
                << " elapsed_ms=" << icpTimer.toc());
        }

        if (registration.hasConverged() == false || score > historyKeyframeFitnessScore)
            return true;
        mapLock.lock();
        if (revision != mapRevision) {
            if (writeDetails) ROS_INFO_STREAM("loop_check stage=stale_snapshot current=" << loopKeyCur);
            return false; // Retry this keyframe against the corrected map on the next wake.
        }
        fitnessScore = score;
        
        if(debugMode) cout<<"found loop, fitness score: "<< fitnessScore<<endl;
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
        if (writeDetails)
            ROS_INFO_STREAM("loop_check stage=queued current=" << loopKeyCur << " previous=" << loopKeyPre);
        return true;
    }

    bool detectLoopClosureDistance(int loopKeyCur, int *closestID)
    {
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
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->points[loopKeyCur], historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        // cout<<copy_cloudKeyPoses6D->points[id].time-cloudKeyPoses6D->points[id].time<<" "<<copy_cloudKeyPoses6D->points[id].time-cloudInfoTime <<endl;
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            
            // A delayed check must use this keyframe's time and only its past.
            if (copy_cloudKeyPoses6D->points[loopKeyCur].time - copy_cloudKeyPoses6D->points[id].time > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum, int lastKey, bool doFiltering = true)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = std::min(static_cast<int>(copy_cloudKeyPoses6D->size()), lastKey + 1);
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
        #pragma omp parallel for num_threads(numberOfCores) if(cloudSize >= 8192)
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
        std::lock_guard<std::recursive_mutex> mapLock(mapDataMutex);
        if (!localizationMode || !mapLoaded || cloudKeyPoses3D->empty()) {
            ROS_WARN("Ignoring initial pose: a loaded localization map is required");
            return;
        }
        const auto& position = pose_msg->pose.pose.position;
        const auto& orientation = pose_msg->pose.pose.orientation;
        const auto& requestedFrame = pose_msg->header.frame_id;
        if (requestedFrame != mapFrame && requestedFrame != "/" + mapFrame) {
            ROS_WARN("Ignoring initial pose outside the loaded map frame");
            return;
        }
        const double quaternionNorm = std::sqrt(orientation.x * orientation.x + orientation.y * orientation.y +
                                                orientation.z * orientation.z + orientation.w * orientation.w);
        if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z) ||
            !std::isfinite(quaternionNorm) || quaternionNorm < 1e-8) {
            ROS_WARN("Ignoring invalid initial pose");
            return;
        }
        ROS_INFO("initial pose received!!");
        std::lock_guard<std::mutex> lock(pose_estimator_mutex);
        poseEstVec.push_back(pose_msg);
        if (relocByRviz()){
            ROS_INFO("Got pose estimate");
            poseGuessFromRvizAvailable = true;
            {
                std::lock_guard<std::mutex> queueLock(queueMutex);
                while (!cloudInfoBuffer.empty()) cloudInfoBuffer.pop();
                while (!lidarOdometryBuffer.empty()) lidarOdometryBuffer.pop();
            }
            if (fixedSubmap) prepareFixedSubmap(rvizInitialPose.translation());
            geometry_msgs::PoseWithCovarianceStamped accepted = *pose_msg;
            accepted.header.frame_id = mapFrame;
            accepted.pose.pose.position.z = rvizInitialPose(2, 3);
            float selected[6];
            Affine3f2Trans(rvizInitialPose, selected);
            accepted.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(selected[0], selected[1], selected[2]);
            pubInitialPoseAccepted.publish(accepted);
        }
    }


    bool relocByRviz() {
        if (!poseEstVec.empty())
        {
            kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            PointType pt;
            vector<int> pointSearchIndLoop;
            vector<float> pointSearchSqDisLoop;

            // added by ACE 2024.4.18
            const auto& pt_rviz_initial = poseEstVec.back()->pose.pose.position;
            pt.x = pt_rviz_initial.x;
            pt.y = pt_rviz_initial.y;
            pt.z = pt_rviz_initial.z;
            std::vector<int> pointSearchIndInitial;
            std::vector<float> pointSearchSqDisInitial;
            if (kdtreeHistoryKeyPoses->nearestKSearch(pt, 1, pointSearchIndInitial, pointSearchSqDisInitial) != 1) {
                poseEstVec.clear();
                ROS_WARN("Initial pose rejected: no map keyframe available");
                return false;
            }
            pt.z = cloudKeyPoses3D->points[pointSearchIndInitial[0]].z;

            rvizPoseType pose_msg = poseEstVec.back();
            const auto& p = pose_msg->pose.pose.position;
            const auto& q = pose_msg->pose.pose.orientation;
            tf::Quaternion q_tf(q.x,q.y,q.z,q.w);
            q_tf.normalize();
            double roll,pitch,yaw;
            tf::Matrix3x3(q_tf).getRPY(roll,pitch,yaw);
            rvizInitialPose = pcl::getTransformation(p.x, p.y, pt.z, (float)roll,(float)pitch,(float)yaw);
            poseEstVec.clear();
            return true;     
        }
        return false;
    }

    void updateInitialGuess()
    {
        
        // only use rviz in loc mode
        if(localizationMode && poseGuessFromRvizAvailable)
        {
            Eigen::Affine3f tCorrect = rvizInitialPose;
            if (frontendInitialization && !frontendReferenceApplied) {
                tCorrect=tCorrect*frontendReference.inverse()*affine_imu_to_odom;
                frontendReferenceApplied=true;
            }

            affine_imu_to_map = tCorrect; 
            affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse(); // for real-time odometry publish
            pcl::getTranslationAndEulerAngles(tCorrect, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
            for(int i = 0;i < 6;i++){
                rvizGuess[i] = transformTobeMapped[i];
                transformBeforeMapped[i] = transformTobeMapped[i];
            }
            printTrans("The reloc pose given by rviz: ",transformTobeMapped);
            poseGuessFromRvizAvailable = false;
            ++localizationGeneration;
            localizationGate.reset();
            initialPoseReceived = true;
            relocSuccess = false;
            tryReloc = true;
            globalPath.poses.clear();
            globalPathFusion.poses.clear();
            globalPathFusionVINS.poses.clear();
            globalPath.header.stamp = timeLidarInfoStamp;
            globalPath.header.frame_id = mapFrame;
            globalPathFusion.header = globalPath.header;
            globalPathFusionVINS.header = globalPath.header;
            pubPath.publish(globalPath);
            pubPathFusion.publish(globalPathFusion);
            pubPathFusionVINS.publish(globalPathFusionVINS);
            // The callback already prepared geometry for this exact selection.
            if (!fixedSubmap) cachedLocalKeyframes.clear();
            return;
        }

        // initialization 
        if (relocSuccess == false && !tryReloc)
        {
            if(!gtAvailable && !useGPS)
            {
                // by default, the 1st localized frame pose is identity(), in fastlio, actually no
                for(int i=0;i<6;i++){
                    transformTobeMapped[i] = initialGuess[i];
                    transformBeforeMapped[i] = initialGuess[i];
                }
                // FAST-LIO already established gravity. Do not discard it by forcing a level LiDAR pose.
                if (!localizationMode && gravityAlignedMap) {
                    float sensorPose[6];
                    Affine3f2Trans(affine_imu_to_odom, sensorPose);
                    transformTobeMapped[0] = sensorPose[0];
                    transformTobeMapped[1] = sensorPose[1];
                    transformBeforeMapped[0] = sensorPose[0];
                    transformBeforeMapped[1] = sensorPose[1];
                }
                affine_imu_to_map = trans2Affine3f(transformTobeMapped);
                if (localizationMode && frontendInitialization && !frontendReferenceApplied) {
                    affine_imu_to_map=affine_imu_to_map*frontendReference.inverse()*affine_imu_to_odom;
                    frontendReferenceApplied=true;
                    Affine3f2Trans(affine_imu_to_map,transformTobeMapped);
                    std::copy(transformTobeMapped,transformTobeMapped+6,transformBeforeMapped);
                }
                affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse();
                cout<<"Todom2map: "<<affine_odom_to_map.matrix()<<endl;
                printTrans("Reloc success by zero initial: ",transformTobeMapped);
                global_estimator->setTgl(affine_odom_to_map.matrix().cast<double>());
                global_estimator->inputGlobalLocPose(cloudInfoTime, affine_imu_to_map.matrix().cast<double>(), 0.1, 4e-4);   
                // cout<<"Relocalized without ground truth or GPS!"<<endl;
                relocSuccess = true;
                return ;
            }
            //  use ground truth for initialization in both modes!
            if(!gpsQueue.empty() && gtAvailable)
            {
                // works when gps odometry is more often than lidar odometry
                
                while(!gpsQueue.empty() && cloudInfoTime - gpsQueue.front().header.stamp.toSec()  > 0.1 )
                {
                    // cout<<setiosflags(ios::fixed)<<setprecision(3)<<gpsQueue.front().header.stamp.toSec() - cloudInfoTime<<endl;
                    mtxGPSqueue.lock();
                    gpsQueue.pop_front();
                    mtxGPSqueue.unlock();
                }                  

                if(gpsQueue.empty() || cloudInfoTime - gpsQueue.front().header.stamp.toSec() < -0.1) return;
                odometryMsgToAffine3f(gpsQueue.front(),affine_imu_to_map);
                Affine3f2Trans(affine_imu_to_map,transformTobeMapped);
                affine_odom_to_map = affine_imu_to_map*affine_imu_to_odom.inverse();
                cout<<"Todom2map: "<<affine_odom_to_map.matrix()<<endl;
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
                    cout<<"Todom2map: "<<affine_odom_to_map.matrix()<<endl;
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

    void installFixedSubmap(const FixedMapSubmap<PointTypePose>::ResultPtr& result)
    {
        submapPending = false;
        cachedLocalKeyframes = result->ids; localSubmapCenter = result->center;
        lidarCloudCornerFromMap = result->corners; lidarCloudSurfFromMap = result->surfaces;
        lidarCloudCornerFromMapDS = result->cornerDS; lidarCloudSurfFromMapDS = result->surfaceDS;
        localCornerTree = result->cornerTree; localSurfaceTree = result->surfaceTree;
        lidarCloudCornerFromMapDSNum = result->cornerDS->size();
        lidarCloudSurfFromMapDSNum = result->surfaceDS->size();
        submapBuilds = result->builds; cacheMisses = result->misses; lastSubmapBuildMs = result->buildMs;
    }

    void prepareFixedSubmap(const Eigen::Vector3f& center)
    {
        fixedSubmap->request(center, ++submapEpoch);
        submapPending = true;
        installFixedSubmap(fixedSubmap->take(submapEpoch, true));
    }

    void extractNearby()
    {
        if (cloudKeyPoses3D->empty() == true) 
            return; 
        const Eigen::Vector3f center(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]);
        if (fixedSubmap) {
            const bool reset = cachedLocalKeyframes.empty();
            if (!reset && !submapPending &&
                (center-localSubmapCenter).norm() >= 2*localizationSubmapUpdateDistance) {
                prepareFixedSubmap(center);
                return;
            }
            if (reset) {
                ++submapEpoch;
                fixedSubmap->request(center, submapEpoch);
                submapPending = true;
            }
            auto result = fixedSubmap->take(submapEpoch, reset ||
                (submapPending && (center-localSubmapCenter).norm() >= 2*localizationSubmapUpdateDistance));
            if (result) {
                // Large motion during construction must not install geometry
                // already outside the reuse budget, even just after waiting.
                if ((center-result->center).norm() >= 2*localizationSubmapUpdateDistance)
                    prepareFixedSubmap(center);
                else installFixedSubmap(result);
            }
            if (!submapPending && !cachedLocalKeyframes.empty() &&
                (center-localSubmapCenter).norm() >= localizationSubmapUpdateDistance) {
                fixedSubmap->request(center, submapEpoch);
                submapPending = true;
            }
            return;
        }
        // Only immutable localization maps may reuse geometry across motion.
        // Clearing membership (reinitialization, failed match, loop correction)
        // forces refresh regardless of the distance from the old anchor.
        if (localizationMode && !mapUpdateEnabled && !temporaryMappingEnabled && useKeyframeCache &&
            !cachedLocalKeyframes.empty() &&
            (center - localSubmapCenter).norm() < localizationSubmapUpdateDistance)
            return;
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        if (!localizationMode || keyposeTreeSize != cloudKeyPoses3D->size()) {
            kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
            keyposeTreeSize = cloudKeyPoses3D->size();
        }
        PointType pt;
        pt.x=transformTobeMapped[3];
        pt.y=transformTobeMapped[4];
        pt.z=transformTobeMapped[5];
        
        kdtreeSurroundingKeyPoses->radiusSearch(pt, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);

        if (pointSearchInd.empty()) {
            lidarCloudCornerFromMapDS->clear();
            lidarCloudSurfFromMapDS->clear();
            lidarCloudCornerFromMapDSNum = lidarCloudSurfFromMapDSNum = 0;
            cachedLocalKeyframes.clear();
            localCornerTree.reset(); localSurfaceTree.reset();
            ROS_WARN_THROTTLE(1, "No local map near current initial pose");
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
        {
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
        localSubmapCenter = center;
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        vector<int> selection;
        selection.reserve(cloudToExtract->size());
        for (const auto& pose : *cloudToExtract) selection.push_back(static_cast<int>(pose.intensity));
        // Duplicate IDs retain legacy weighting; sorting stabilizes membership comparisons.
        std::sort(selection.begin(), selection.end());
        if (useKeyframeCache && selection == cachedLocalKeyframes) return;
        cachedLocalKeyframes = selection;
        ++submapBuilds;
        // fuse the map
        lidarCloudCornerFromMap->clear();
        lidarCloudSurfFromMap->clear(); 
        // TicToc transPC;
        for (int thisKeyInd : selection) {

            if (!useKeyframeCache) {
                ++cacheMisses;
                *lidarCloudCornerFromMap += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                *lidarCloudSurfFromMap += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                continue;
            }
            auto cached = lidarCloudMapContainer.find(thisKeyInd);
            if (cached == lidarCloudMapContainer.end()) {
                ++cacheMisses;
                auto corner = transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                auto surface = transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                const size_t bytes = (corner->size() + surface->size()) * sizeof(PointType);
                while (!lidarCloudMapContainer.empty() &&
                       (lidarCloudMapContainer.size() >= static_cast<size_t>(keyframeCacheMaxEntries) ||
                        keyframeCacheBytes + bytes > static_cast<size_t>(keyframeCacheMaxBytes))) {
                    auto oldest = std::min_element(keyframeCacheAccess.begin(), keyframeCacheAccess.end(),
                        [](const auto& a, const auto& b) { return a.second < b.second; });
                    auto entry = lidarCloudMapContainer.find(oldest->first);
                    keyframeCacheBytes -= (entry->second.first.size() + entry->second.second.size()) * sizeof(PointType);
                    lidarCloudMapContainer.erase(entry); keyframeCacheAccess.erase(oldest);
                }
                if (bytes > static_cast<size_t>(keyframeCacheMaxBytes)) {
                    *lidarCloudCornerFromMap += *corner; *lidarCloudSurfFromMap += *surface;
                    continue;
                }
                cached = lidarCloudMapContainer.emplace(thisKeyInd, std::make_pair(std::move(*corner), std::move(*surface))).first;
                keyframeCacheBytes += bytes;
            }
            keyframeCacheAccess[thisKeyInd] = ++cacheSequence;
            *lidarCloudCornerFromMap += cached->second.first;
            *lidarCloudSurfFromMap += cached->second.second;
        }
        
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(lidarCloudCornerFromMap);
        downSizeFilterCorner.filter(*lidarCloudCornerFromMapDS);
        lidarCloudCornerFromMapDSNum = lidarCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(lidarCloudSurfFromMap);
        downSizeFilterSurf.filter(*lidarCloudSurfFromMapDS);
        lidarCloudSurfFromMapDSNum = lidarCloudSurfFromMapDS->size();
        localCornerTree.reset(new pcl::KdTreeFLANN<PointType>());
        localSurfaceTree.reset(new pcl::KdTreeFLANN<PointType>());
        if (!lidarCloudCornerFromMapDS->empty()) localCornerTree->setInputCloud(lidarCloudCornerFromMapDS);
        if (!lidarCloudSurfFromMapDS->empty()) localSurfaceTree->setInputCloud(lidarCloudSurfFromMapDS);
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
        // cout<<"After sampling: "<<lidarCloudCornerLastDSNum<<endl;
        lidarCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(lidarCloudSurfLast);
        downSizeFilterSurf.filter(*lidarCloudSurfLastDS);
        lidarCloudSurfLastDSNum = lidarCloudSurfLastDS->size();
    }

    void scan2MapOptimization()
    {
        // no clouds nearby
        if (cloudKeyPoses3D->empty() || lidarCloudCornerFromMapDS->empty() || lidarCloudSurfFromMapDS->empty())
        {
            localizationGate.missing();
            return;
        }
        
        // cout<<"corner, surf points: "<<lidarCloudCornerLastDSNum<<" "<<lidarCloudSurfLastDSNum<<endl;
        if (lidarCloudCornerLastDSNum > edgeFeatureMinValidNum && lidarCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // get a copy of those clouds might lower down speed
            if (!matcher) matcher.reset(new LOAMmapping(*this));
            LOAMmapping& LM = *matcher;
            LM.traceEnabled = writeDetails;
            LM.reset(lidarCloudCornerLastDS, lidarCloudSurfLastDS, lidarCloudCornerFromMapDS,
                     lidarCloudSurfFromMapDS, affine_imu_to_map, localCornerTree, localSurfaceTree);
            LM.match();
            if (writeDetails) {
                for (const auto& stats : LM.iterationTrace) {
                    registrationFile << timeLidarInfoStamp.toNSec() << ',' << stats.iteration << ','
                        << stats.correspondences << ',' << stats.degenerate << ',' << std::setprecision(12)
                        << stats.min_eigen << ',' << stats.residual << ',' << stats.raw_rotation << ',' << stats.raw_translation;
                    for (int i = 0; i < 6; ++i) registrationFile << ',' << stats.before(i);
                    for (int i = 0; i < 6; ++i) registrationFile << ',' << stats.step(i);
                    for (int i = 0; i < 6; ++i) registrationFile << ',' << LM.firstEigenvalues(i);
                    for (int i = 0; i < 6; ++i) registrationFile << ',' << LM.weakestDirection(i);
                    registrationFile << '\n';
                }
            }
            lastInlierRatio = LM.inlier_ratio2;
            lastRegistrationError = LM.regiError;
            matchingValid = localizationGate.accept(gatePrediction, LM.affine_out, affine_imu_to_odom,
                                                     lastInlierRatio, lastRegistrationError);
            
            // for relocalization in loc mode: only needed when used in actual world
            if (matchingValid && tryReloc == true)
            {
                ROS_INFO_STREAM("At time "<< cloudInfoTime - rosTimeStart <<" sec, relocalization succeeds!");
                relocSuccess = true;
                tryReloc = false;
                transformUpdate(); // for giving fusion pose immediate map_odom
            }

            if (relocSuccess == true && localizationMode == true)
            {
                // for entering TMM: 
                //inlier_ratio2 is more sensitive to map extension, and will suffice
                if (temporaryMappingEnabled && LM.inlier_ratio2 < startTemporaryMappingInlierRatioThre && !temporaryMappingMode)
                {
                    ROS_INFO_STREAM("At time "<< cloudInfoTime - rosTimeStart <<" sec, Entering temporary mapping mode due to poor mapping performace");
                    ROS_INFO_STREAM("Inlier ratio2: "<< LM.inlier_ratio2);                        
                    temporaryMappingMode = true; // here is the case for outdated map
                    startTemporaryMappingIndex = temporaryCloudKeyPoses3D->size();
                    frameTobeAbandoned = true;
                    TMMcount++;                   
                }
                
                // more strict to exit TMM for map updating
                if (LM.inlier_ratio2 > exitTemporaryMappingInlierRatioThre && int(temporaryCloudKeyPoses3D->size()) > slidingWindowSize + 10 && temporaryMappingMode == true)
                {
                    correctedPose = LM.affine_out;// notice: the correction cannot be simply the correction for last keyframe!
                    affine_imu_to_map = LM.affine_out;
                    // LM.getTransformation(transformTobeMapped); // don't change it here, need original one as odom factor
                    goodToMergeMap = true;
                    cout<<"Now it is okay to merge the temporary map"<<endl;
                }
            }
            // only correct pose from fastlio when mapping quality is good
            // if (temporaryMappingMode == false && LM.isDegenerate == false)  // worse
            // when building a map, it is better to fuse LIO and global matching
            if ((!localizationMode || matchingValid) && !temporaryMappingMode) 
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
            // cout<<"inlierRatio2: "<<LM.inlier_ratio2<<endl;        
        } 
        else 
        {
            localizationGate.missing();
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
        if (localizationMode)
        {
            if (temporaryCloudKeyPoses6D->empty() == true ) return isIndoorJudgement(); 
            transStart = pclPointToAffine3f(temporaryCloudKeyPoses6D->back());  
        }            
        else
            transStart = pclPointToAffine3f(cloudKeyPoses6D->back());

        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        float dist = sqrt(x*x + y*y + z*z);
        if (dist > surroundingkeyframeAddingDistThreshold) // for paranomic lidar angle threshold is not needed
            return isIndoorJudgement();
        // cout<<"dist: "<<dist<<endl;
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
        // cout<<"ratio: " <<ratio<<" count: "<<count<<endl;
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
            // cout<<gpsQueue.size()<<" "<<cloudKeyPoses3D->size()<<" "<<useGPS<<endl;
            return;
        }
            
        if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
        {
            // cout<<"key pose distance is within 5 meter"<<endl;
            return;
        }
            

        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
        {
            // cout<<"pose cov is still very small, no need for gps correction"<<endl;
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
                // cout<<"the oldest message is too old, pop it out"<<endl;
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > cloudKeyPoses6D->points[cloudSize-1].time + 0.1)
            {
                // cout<<" all messages are too new" << endl;
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
                    cout<<"too much difference between gps and lidar slam; x diff: "<<abs(gps_x - cloudKeyPoses6D->points[cloudSize-1].x)<<
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
            if (writeDetails)
                ROS_INFO_STREAM("loop_check stage=added_to_graph current=" << indexFrom << " previous=" << indexTo);
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

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        if (aLoopIsClosed == true)
        {
            ++mapRevision;
            if (writeDetails)
                ROS_INFO_STREAM("loop_check stage=optimized revision=" << mapRevision << " t=" << cloudInfoTime-rosTimeStart);
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
        // cout << "****************************************************" << endl;
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

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
        // cout<<"x cov: "<< poseCovariance(3,3)<< " y cov: "<<poseCovariance(4,4)<<endl;

        // cout<<"Before opt: "<<transformTobeMapped[3]<<endl;;
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

        // cout<<"After opt: "<< transformTobeMapped[3]<<endl;
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
            keyframeCacheAccess.clear();
            keyframeCacheBytes = 0;
            globalDisplayCached = false;
            cachedLocalKeyframes.clear();
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

    void publishLocalizationState(const nav_msgs::Odometry& frontend)
    {
        social_nav_localization::LocalizationState state;
        state.header.stamp = timeLidarInfoStamp;
        state.header.frame_id = localFrame;
        state.child_frame_id = lidarFrame;
        // The frontend reports the same physical LiDAR in its continuous world.
        // In this profile "lidar" and "rslidar" are explicit names for that sensor.
        state.pose = frontend.pose.pose;
        state.has_covariance = false; // Frontend LiDAR covariance is not supplied.
        const bool valid = matchingValid && std::isfinite(lastRegistrationError) && lastRegistrationError >= 0;
        state.health = valid ? state.VALID : state.INVALID;
        state.status = localizationMode ? (valid ? "TRACKING" : "LOST") : (valid ? "MAPPING" : "INITIALIZING");
        state.reason = valid ? "" : (localizationGate.reason.empty() ?
            "map_registration_unavailable_or_failed" : localizationGate.reason);
        state.source = stationaryImuAssumption ? "roll_fast_lio_stationary_assumption" : "roll_fast_lio";
        state.session_id = localizationSession;
        state.generation = localizationGeneration;
        state.has_map_pose = valid;
        state.map_frame = mapFrame;
        state.map_revision = mapRevision;
        state.map_pose.position.x = transformTobeMapped[3];
        state.map_pose.position.y = transformTobeMapped[4];
        state.map_pose.position.z = transformTobeMapped[5];
        state.map_pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(
            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        state.has_matching_quality = std::isfinite(lastRegistrationError) && lastRegistrationError >= 0;
        state.inlier_ratio = lastInlierRatio;
        state.registration_error_m = state.has_matching_quality ? lastRegistrationError : 0;
        pubLocalizationState.publish(state);
        if (valid && publishLocalizationTF) {
            tf::Transform localFromLidar, mapFromLidar;
            tf::poseMsgToTF(state.pose, localFromLidar);
            tf::poseMsgToTF(state.map_pose, mapFromLidar);
            static tf::TransformBroadcaster broadcaster;
            broadcaster.sendTransform(tf::StampedTransform(mapFromLidar * localFromLidar.inverse(),
                state.header.stamp, mapFrame, localFrame));
            broadcaster.sendTransform(tf::StampedTransform(localFromLidar,
                state.header.stamp, localFrame, lidarFrame));
        }
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
        // cout<<transformTobeMapped[3]<<" "<<transformTobeMapped[4]<<" "<<endl;
        lidarOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        lidarOdometryROS.child_frame_id = lidarFrame;
        if (localizationMode && !matchingValid)
            for (int i = 0; i < 6; ++i) lidarOdometryROS.pose.covariance[7*i] = 1e6;
        pubLidarOdometryGlobal.publish(lidarOdometryROS);
        const auto& p = lidarOdometryROS.pose.pose.position;
        const auto& q = lidarOdometryROS.pose.pose.orientation;
        if (writeDetails) trajectoryFile << timeLidarInfoStamp.toSec() << ' ' << p.x << ' ' << p.y << ' ' << p.z
                       << ' ' << q.x << ' ' << q.y << ' ' << q.z << ' ' << q.w << '\n';
        // globalOdometry.push_back(lidarOdometryROS);
    }

    void publishLocalMap()
    {
        globalPath.header.stamp = timeLidarInfoStamp;
        globalPath.header.frame_id = mapFrame;
        if (pubPath.getNumSubscribers()) pubPath.publish(globalPath);
        if (pubRecentKeyFrames.getNumSubscribers() == 0) return;
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
        
        // added by ACE 2024.4.18
        pcl::VoxelGrid<PointType> downSizeFilterLocalMapKeyFrames; // for local map visualization
        downSizeFilterLocalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterLocalMapKeyFrames.setInputCloud(cloudLocal);
        downSizeFilterLocalMapKeyFrames.filter(*cloudLocal);

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

            if(debugMode) std::cout << "current pid: " << current_pid << std::endl;
            if(debugMode) std::cout << "cpu usage ratio: " << cpu_usage_ratio * 100 << "%" << std::endl;
            if(debugMode) std::cout << "memory usage: " << memory_usage << "MB" << std::endl;
            rate.sleep();
        }
        
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roll");

    try {
    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);
    std::thread mappingThread{&mapOptimization::run,&MO};
    std::thread getCPUinfoThread(&mapOptimization::getCPUinfoThread, &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();
    mappingThread.join();
    getCPUinfoThread.join();
    if (!MO.localizationMode && MO.saveKeyframeMap && !MO.doneSavingMap) {
        roll::save_mapRequest req; roll::save_mapResponse res;
        MO.saveMapService(req, res);
        if (!res.success) return 2;
    }
    return 0;
    } catch (const std::exception& error) {
        ROS_FATAL("ROLL initialization/runtime error: %s", error.what());
        return 1;
    }
}
