
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
