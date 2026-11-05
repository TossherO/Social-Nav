
        nh.param<double>("roll/globalMatchingRate", globalMatchingRate, 10.0);
#ifndef SOCIAL_NAV_NATIVE_ROLL
        std::string matchingBackend;nh.param<std::string>("roll/matchingBackend",matchingBackend,"cpu");
        if(matchingBackend!="cpu")throw std::runtime_error("ROLL CUDA matching requires the direct native runtime; ROS diagnostic supports cpu only");
#endif

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
            throw std::runtime_error("Invalid sensor type: "+sensorStr);
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
