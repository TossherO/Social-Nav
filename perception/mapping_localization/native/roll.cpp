// ROLL C ABI. The solver/submap/pose-graph functions are shared with its ROS shell.
#include "scan_features.h"
#include "stages.h"
#include <omp.h>
#include "LOAMmapping.h"
#include "MapStorage.h"
#include "FixedMapSubmap.h"
#include "LocalizationGate.h"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <atomic>
#define ROS_INFO_STREAM(...) do{}while(0)
#define ROS_WARN_THROTTLE(...) do{}while(0)
using namespace gtsam;
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

struct NativeStamp {double seconds=0;uint64_t toNSec()const{return llround(seconds*1e9);}double toSec()const{return seconds;}};
class NativeRoll:public ParamServer {
public:
    // for trajectory alignment

    long long cnt = 0;

    int current_pid ;//or you can set a outside program pid
    float cpu_usage_ratio = 0;
    float memory_usage = 0;

    string logTime = "";

    ofstream pose_log_file;
    float fitnessScore = 0;
    ofstream gps_file;

    Eigen::Affine3f lastOdometryPose;

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
    Eigen::Affine3f rvizInitialPose;

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;




    

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
    
    double cloudInfoTime;

    float transformTobeMapped[6];
    
    std::mutex mtx;
    std::mutex queueMutex;
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
    bool waitForInitialPose = false, initialPoseReceived = false;
    bool frontendInitialization = false, frontendReferenceReady = false, frontendReferenceApplied = false;
    Eigen::Affine3f frontendReference = Eigen::Affine3f::Identity();
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
    
    


    bool poseGuessFromRvizAvailable = false;
    float rvizGuess[6];

    
    pcl::PointCloud<PointType>::Ptr lidarCloudRaw; 

// one keyframe per 1m, considering "0.5m & 0.1 deg"drift per 100m, for every keyframe
    // translation: 0.5/100=5e-3 m
    // rotation: 0.1/57.3/100 = 2e-5
    // Variances, not Covariances!!!
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr corrNoise; // used for loop closusre and prior noise in merging


    NativeStamp timeLidarInfoStamp;
    Features features;
    std::atomic<bool> stop{false};
    std::thread loopWorker;
    std::condition_variable loopReady;
    std::mutex loopMutex;
    std::atomic<int> loopProcessed{0};
    uint64_t loopWake=0;
    bool havePose=false;
    SnRollStages stages{};
    double previousStamp=-INFINITY,lastLoopStamp=-INFINITY;
    void updatePath(const PointTypePose&){} // Paths are exported by the runtime display consumer.
    void transformUpdate(){affine_odom_to_map=affine_imu_to_map*affine_imu_to_odom.inverse();}
    void addGPSFactor(){} // Constructor rejects GPS/GT/secondary fusion for this deployment.
    #include "MapMath.inc"

    explicit NativeRoll(const YAML::Node& c):ParamServer(c),features(c) {
        if(useGPS||gtAvailable||enableSecondaryFusion||temporaryMappingEnabled||mapUpdateEnabled)
            throw std::invalid_argument("Native ROLL supports the current immutable-map / LiDAR mapping configuration");
        string backend;nh.param("roll/matchingBackend",backend,"cpu");
        if(backend!="cpu"&&backend!="cuda")throw std::invalid_argument("ROLL matchingBackend must be cpu or cuda");
        matcher.reset(new LOAMmapping(*this));
        if(backend=="cuda"){
#ifdef SOCIAL_NAV_LIO_CUDA
            int device,threads;float cell;bool corners,plane_fit;nh.param("roll/matchingCudaDevice",device,0);
            nh.param("roll/matchingCudaCellSize",cell,.5f);nh.param("roll/matchingCudaCorners",corners,false);
            nh.param("roll/matchingCudaPlaneFit",plane_fit,false);
            nh.param("roll/matchingCudaThreads",threads,1);
            if(threads<1)throw std::invalid_argument("ROLL matchingCudaThreads must be positive");matcher->matchingThreads=threads;
            matcher->enableCuda(device,cell,corners,plane_fit);stages.backend=1;
#else
            throw std::invalid_argument("ROLL CUDA matching requires SOCIAL_NAV_BUILD_LIO_CUDA=ON");
#endif
        }
        resetISAM();
        auto noise=[](const vector<float>& v){return noiseModel::Diagonal::Variances((Vector(6)<<v[0],v[1],v[2],v[3],v[4],v[5]).finished());};
        odometryNoise=noise(odom_variance);corrNoise=noise(loop_variance);
        affine_imu_to_map=affine_imu_to_odom=affine_odom_to_map=Eigen::Affine3f::Identity();
        submap.reset(new pcl::PointCloud<PointType>);
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>);cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>);
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>);copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>);
        temporaryCloudKeyPoses3D.reset(new pcl::PointCloud<PointType>);temporaryCloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>);
        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>);kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>);
        lidarCloudRaw.reset(new pcl::PointCloud<PointType>);lidarCloudCornerLast.reset(new pcl::PointCloud<PointType>);
        lidarCloudSurfLast.reset(new pcl::PointCloud<PointType>);lidarCloudCornerLastDS.reset(new pcl::PointCloud<PointType>);
        lidarCloudSurfLastDS.reset(new pcl::PointCloud<PointType>);lidarCloudCornerFromMap.reset(new pcl::PointCloud<PointType>);
        lidarCloudSurfFromMap.reset(new pcl::PointCloud<PointType>);lidarCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>);
        lidarCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>);
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize,mappingCornerLeafSize,mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize,mappingSurfLeafSize,mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize,mappingSurfLeafSize,mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity,surroundingKeyframeDensity,surroundingKeyframeDensity);
        std::fill(transformTobeMapped,transformTobeMapped+6,0);
        std::fill(transformBeforeMapped,transformBeforeMapped+6,0);
        if(localizationMode) {
            roll_storage::load(saveKeyframeMapDirectory,*cloudKeyPoses6D,*cloudKeyPoses3D,cornerCloudKeyFrames,surfCloudKeyFrames,isIndoorKeyframe);
            mapLoaded=true;
            bool async=true;nh.param("roll/localizationAsyncSubmap",async,true);
            if(async&&useKeyframeCache&&localizationSubmapUpdateDistance>0)
                fixedSubmap.reset(new FixedMapSubmap<PointTypePose>(*cloudKeyPoses6D,cornerCloudKeyFrames,surfCloudKeyFrames,
                    surroundingKeyframeSearchRadius,surroundingKeyframeDensity,mappingCornerLeafSize,mappingSurfLeafSize,
                    keyframeCacheMaxEntries,keyframeCacheMaxBytes));
        }
        auto coverage=c["roll"]["localization_coverage"];
        if(coverage&&coverage["regions_xy"]) {
            if(!localizationMode||coverage["frame_id"].as<string>()!=mapFrame)throw std::invalid_argument("Invalid map coverage frame/mode");
            localizationGate.enabled=true;
            for(auto region:coverage["regions_xy"]) {vector<Eigen::Vector2f> vertices;
                for(auto xy:region)vertices.emplace_back(xy[0].as<float>(),xy[1].as<float>());
                if(vertices.size()<3)throw std::invalid_argument("Invalid coverage polygon");localizationGate.regions.push_back(vertices);}
            auto s=c["roll"]["coverageGate"];
            auto get=[&](const char* name,double& value){if(s[name])value=s[name].as<double>();};
            get("reentryMargin",localizationGate.reentryMargin);get("minInlier",localizationGate.minInlier);
            get("maxResidual",localizationGate.maxResidual);get("maxStep",localizationGate.maxStep);
            get("maxAngle",localizationGate.maxAngle);get("maxGravity",localizationGate.maxGravity);
            get("consistencyDistance",localizationGate.consistencyDistance);get("consistencyAngle",localizationGate.consistencyAngle);
            if(s["confirmationMatches"])localizationGate.confirmationMatches=s["confirmationMatches"].as<int>();
        }
        if(!localizationMode&&loopClosureEnableFlag)loopWorker=std::thread([this]{
            int next=0;uint64_t seen=0;
            while(!stop){std::unique_lock<std::mutex> lock(loopMutex);
                loopReady.wait(lock,[&]{return stop||loopWake!=seen;});seen=loopWake;lock.unlock();
                while(!stop&&performLoopClosure(next))++next;
                loopProcessed=next;loopReady.notify_all();
            }
        });
    }
    ~NativeRoll(){stop=true;loopReady.notify_all();if(loopWorker.joinable())loopWorker.join();delete isam;}
    int step(const float* cloud,size_t n,const double* pose,const double* reference,double stamp,double* out) {
        int backend=stages.backend;stages={};stages.backend=backend;
        matcher->beginFrame();
        if(!std::isfinite(stamp)||stamp<=previousStamp)throw std::invalid_argument("Reversed ROLL scan time");
        previousStamp=stamp;timeLidarInfoStamp.seconds=stamp;cloudInfoTime=stamp;
        if(n<1000){std::fill(out,out+19,0.);return 2;} // Original feature edge emits no observation.
        if(rosTimeStart<0)rosTimeStart=stamp;
        Eigen::Affine3f frontend;
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)frontend.matrix()(i,j)=pose[i*4+j];
        auto raw=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        raw->reserve(n);for(size_t i=0;i<n;++i){PointType p;p.x=cloud[4*i];p.y=cloud[4*i+1];p.z=cloud[4*i+2];p.intensity=cloud[4*i+3];raw->push_back(p);}
        double stage_begin=omp_get_wtime();features.compute(raw,lidarCloudCornerLast,lidarCloudSurfLast);lidarCloudRaw=raw;
        stages.feature_ms=(omp_get_wtime()-stage_begin)*1000;
        std::lock_guard<std::recursive_mutex> lock(mapDataMutex);
        affine_imu_to_odom=frontend;
        if(!havePose){
            affine_imu_to_map=trans2Affine3f(initialGuess);
            if(localizationMode){Eigen::Affine3f ref;for(int i=0;i<4;++i)for(int j=0;j<4;++j)ref.matrix()(i,j)=reference[4*i+j];
                affine_imu_to_map=affine_imu_to_map*ref.inverse()*frontend;}
            else if(gravityAlignedMap){float f[6];Affine3f2Trans(frontend,f);
                float first[6];std::copy(initialGuess.begin(),initialGuess.end(),first);first[0]=f[0];first[1]=f[1];
                affine_imu_to_map=trans2Affine3f(first);}
            affine_odom_to_map=affine_imu_to_map*frontend.inverse();havePose=true;relocSuccess=true;
        }else affine_imu_to_map=affine_odom_to_map*frontend;
        Affine3f2Trans(affine_imu_to_map,transformTobeMapped);std::copy(transformTobeMapped,transformTobeMapped+6,transformBeforeMapped);
        gatePrediction=affine_imu_to_map;matchingValid=false;lastInlierRatio=0;lastRegistrationError=-1;
        if(localizationGate.begin(gatePrediction)){
            stage_begin=omp_get_wtime();extractNearby();stages.submap_ms=(omp_get_wtime()-stage_begin)*1000;
            stage_begin=omp_get_wtime();downsampleCurrentScan();stages.downsample_ms=(omp_get_wtime()-stage_begin)*1000;
            stage_begin=omp_get_wtime();scan2MapOptimization();stages.matching_ms=(omp_get_wtime()-stage_begin)*1000;
            stages.corner_ms=matcher->cornerTime;stages.surface_ms=matcher->surfTime;stages.linear_solve_ms=matcher->solveTime;
            stages.iterations=matcher->ranMatch?std::min(matcher->optIteration,matcher->iterCount+1):0;stages.fallbacks=matcher->cudaFallbacks;
#ifdef SOCIAL_NAV_LIO_CUDA
            if(backend){auto s=matcher->cudaStats();stages.map_upload_ms=s.map_ms;stages.query_device_ms=s.query_device_ms;
                stages.h2d_bytes=s.h2d_bytes;stages.d2h_bytes=s.d2h_bytes;}
#endif
            stage_begin=omp_get_wtime();
            if(!localizationMode){saveKeyFramesAndFactor();correctPoses();}
            // Immutable fixed submaps already refresh when the predicted pose
            // moves outside their reuse budget. Invalid registration must not
            // synchronously rebuild the same geometry on every subsequent scan.
            else if(!matchingValid && !fixedSubmap)cachedLocalKeyframes.clear();
            stages.graph_ms=(omp_get_wtime()-stage_begin)*1000;}
        affine_odom_to_map=affine_imu_to_map*frontend.inverse();
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)out[4*i+j]=affine_imu_to_map.matrix()(i,j);
        out[16]=lastInlierRatio;out[17]=lastRegistrationError;out[18]=mapRevision;
        if(!localizationMode&&stamp-lastLoopStamp>=1./loopClosureFrequency){
            {std::lock_guard<std::mutex> wake(loopMutex);++loopWake;}lastLoopStamp=stamp;loopReady.notify_one();}
        return matchingValid?1:0;
    }
    int stage_stats(SnRollStages& out){out=stages;return 0;}
    void save(const char* directory){if(localizationMode)throw std::invalid_argument("Localization prior is immutable");
        if(loopWorker.joinable()){
            int expected;{std::lock_guard<std::recursive_mutex> lock(mapDataMutex);expected=cloudKeyPoses3D->size();}
            {std::lock_guard<std::mutex> lock(loopMutex);++loopWake;}loopReady.notify_all();
            std::unique_lock<std::mutex> lock(loopMutex);
            if(!loopReady.wait_for(lock,std::chrono::seconds(30),[&]{return loopProcessed>=expected;}))
                throw std::runtime_error("Loop closure did not drain before map export");
        }
        std::lock_guard<std::recursive_mutex> lock(mapDataMutex);
        if(!loopIndexQueue.empty()){
            addLoopFactor();isam->update(gtSAMgraph,initialEstimate);isam->update();
            for(int i=0;i<5;++i)isam->update();
            gtSAMgraph.resize(0);initialEstimate.clear();isamCurrentEstimate=isam->calculateEstimate();++mapRevision;correctPoses();
        }
        roll_storage::save(directory,*cloudKeyPoses6D,cornerCloudKeyFrames,surfCloudKeyFrames,isIndoorKeyframe,globalMapVisualizationLeafSize,mapFrame);}
};
static thread_local string rollError;
extern "C" const char* sn_roll_error(){return rollError.c_str();}
extern "C" void* sn_roll_create(const char* path){try{return new NativeRoll(YAML::LoadFile(path));}catch(const exception& e){rollError=e.what();return nullptr;}}
extern "C" void sn_roll_destroy(void* p){delete static_cast<NativeRoll*>(p);}
extern "C" int sn_roll_step(void* p,const float* cloud,size_t n,const double* pose,const double* reference,double stamp,double* out){
    try{return static_cast<NativeRoll*>(p)->step(cloud,n,pose,reference,stamp,out);}catch(const exception& e){rollError=e.what();return -1;}}
extern "C" int sn_roll_save(void* p,const char* dir){try{static_cast<NativeRoll*>(p)->save(dir);return 0;}catch(const exception& e){rollError=e.what();return -1;}}
extern "C" int sn_roll_stages(void* p,SnRollStages* out){return static_cast<NativeRoll*>(p)->stage_stats(*out);}
extern "C" int sn_roll_gate_diagnostics(void* p,double* out){
    const auto& gate=static_cast<NativeRoll*>(p)->localizationGate;
    out[0]=gate.rejectionMask;out[1]=gate.candidateClearance;out[2]=gate.correctionStep;
    out[3]=gate.correctionAngle;out[4]=gate.gravityAngle;return 0;
}
