// Shared ROLL registration, submap and pose-graph implementation.
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
#ifndef SOCIAL_NAV_NATIVE_ROLL
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLidarInfoStamp, mapFrame);
#endif
        // ICP only accesses the already assembled snapshot clouds.
        mapLock.unlock();

        // ICP Settings
        pcl::IterativeClosestPoint<PointType, PointType> registration;
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
#ifndef SOCIAL_NAV_NATIVE_ROLL
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, registration.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLidarInfoStamp, mapFrame);
        }
#endif

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
                     lidarCloudSurfFromMapDS, affine_imu_to_map, localCornerTree, localSurfaceTree,submapBuilds);
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

#ifndef SOCIAL_NAV_NATIVE_ROLL
                if (goodToMergeMap) // reset for the frame of merging
                    global_estimator->resetOptimization(LM.affine_out.matrix().cast<double>());
                else 
                    // radian error is proportional to meter, 1/250
                    global_estimator->inputGlobalLocPose(cloudInfoTime, LM.affine_out.matrix().cast<double>(), 0.1, 4e-4);
#endif             
                
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
#ifndef SOCIAL_NAV_NATIVE_ROLL
            globalPath.poses.clear();
#endif
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
