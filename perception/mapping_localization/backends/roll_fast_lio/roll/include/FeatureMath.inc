        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*lidarCloudIn, *lidarCloudIn, indices);




        int cloudSize = lidarCloudIn->points.size();

        if (cloudSize < 1000)
        {
            ROS_WARN("Empty cloud or points too few %d",cloudSize);
            return;
        } 

        // if (cloudSize > 2*N_SCAN*Horizon_SCAN)
        // {
            // ROS_WARN("Points too many");
            // pcl::io::savePCDFileBinary(saveMapDirectory + "/big_cloud.pcd", *lidarCloudIn);
        // }
        float startOri = -atan2(lidarCloudIn->points[0].y, lidarCloudIn->points[0].x);
        float endOri = -atan2(lidarCloudIn->points[cloudSize - 1].y,
                            lidarCloudIn->points[cloudSize - 1].x) +
                    2 * M_PI;

        if (endOri - startOri > 3 * M_PI)
        {
            endOri -= 2 * M_PI;
        }
        else if (endOri - startOri < M_PI)
        {
            endOri += 2 * M_PI;
        }


        bool halfPassed = false;
        int count = cloudSize;
        PointType point;
        std::vector<pcl::PointCloud<PointType>> lidarCloudScans(N_SCAN);
        
        
        for (int i = 0; i < cloudSize; i++)
        {
            point.x = lidarCloudIn->points[i].x;
            point.y = lidarCloudIn->points[i].y;
            point.z = lidarCloudIn->points[i].z;

            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (N_SCAN == 16)
            {
                scanID = int((angle + 15) / 2 + 0.5);
                if (scanID > (N_SCAN - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (N_SCAN == 32)
            {
                scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
                if (scanID > (N_SCAN - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (N_SCAN == 64)
            {   
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = N_SCAN / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                // use [0 50]  > 50 remove outlies 
                if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else
            {
                printf("wrong scan number\n");
                ROS_BREAK();
            }


            float ori = -atan2(point.y, point.x);
            if (!halfPassed)
            { 
                if (ori < startOri - M_PI / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > startOri + M_PI * 3 / 2)
                {
                    ori -= 2 * M_PI;
                }

                if (ori - startOri > M_PI)
                {
                    halfPassed = true;
                }
            }
            else
            {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > endOri + M_PI / 2)
                {
                    ori -= 2 * M_PI;
                }
            }
        
                float relTime = (ori - startOri) / (endOri - startOri);
                point.intensity = scanID + scanPeriod * relTime;
                lidarCloudScans[scanID].push_back(point); 

        }
        // if(debugMode) printf("Before projection, points size: %d \n", cloudSize);

        pcl::PointCloud<PointType>::Ptr lidarCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < N_SCAN; i++)
        { 
            scanStartInd[i] = lidarCloud->size() + 5;
            *lidarCloud += lidarCloudScans[i];
            scanEndInd[i] = lidarCloud->size() - 6;
        }

        cloudSize = lidarCloud->size();
        if (cloudSize < 12) return;
        if (static_cast<size_t>(cloudSize) > cloudCurvature.size()) {
            cloudCurvature.resize(cloudSize);
            cloudNeighborPicked.resize(cloudSize);
            cloudLabel.resize(cloudSize);
            cloudSmoothness.resize(cloudSize);
        }
        // if(debugMode)  cout<<"After projection, point size: "<<lidarCloud->size()<<endl;
        // printf("prepare time %f \n", t_prepare.toc());

        for (int i = 5; i < cloudSize - 5; i++)
        { 
            float diffX = lidarCloud->points[i - 5].x + lidarCloud->points[i - 4].x + lidarCloud->points[i - 3].x + lidarCloud->points[i - 2].x + lidarCloud->points[i - 1].x - 10 * lidarCloud->points[i].x + lidarCloud->points[i + 1].x + lidarCloud->points[i + 2].x + lidarCloud->points[i + 3].x + lidarCloud->points[i + 4].x + lidarCloud->points[i + 5].x;
            float diffY = lidarCloud->points[i - 5].y + lidarCloud->points[i - 4].y + lidarCloud->points[i - 3].y + lidarCloud->points[i - 2].y + lidarCloud->points[i - 1].y - 10 * lidarCloud->points[i].y + lidarCloud->points[i + 1].y + lidarCloud->points[i + 2].y + lidarCloud->points[i + 3].y + lidarCloud->points[i + 4].y + lidarCloud->points[i + 5].y;
            float diffZ = lidarCloud->points[i - 5].z + lidarCloud->points[i - 4].z + lidarCloud->points[i - 3].z + lidarCloud->points[i - 2].z + lidarCloud->points[i - 1].z - 10 * lidarCloud->points[i].z + lidarCloud->points[i + 1].z + lidarCloud->points[i + 2].z + lidarCloud->points[i + 3].z + lidarCloud->points[i + 4].z + lidarCloud->points[i + 5].z;

            cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
            cloudSmoothness[i].ind = i;
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
        }
        // cout<<"smoothness calculated"<<endl;

        markBadPoints(lidarCloud);
        // if(debugMode) cout<<"After removing bad points, point size: "<<lidarCloud->size()<<endl;
        TicToc t_pts;

        pcl::PointCloud<PointType>::Ptr cornerCloudSharp(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr  cornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr  surfaceCloudFlat(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr  surfaceCloud(new pcl::PointCloud<PointType>());


        float t_q_sort = 0;
        float t_filter = 0;
        for (int i = 0; i < N_SCAN; i++)
        {
            
            if ( i % downsampleRate != 0) continue;
            if( scanEndInd[i] - scanStartInd[i] < 6)
                continue;
            pcl::PointCloud<PointType>::Ptr surfaceCloudTmp(new pcl::PointCloud<PointType>);
            for (int j = 0; j < 6; j++)
            {
                int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
                int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

                TicToc t_tmp;
                std::sort (cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep + 1, by_value());
                t_q_sort += t_tmp.toc();

                int largestPickedNum = 0;
                
                for (int k = ep; k >= sp; k--)// cout<<"start filtering"<<endl;
                {
                    int ind = cloudSmoothness[k].ind; 

                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 2)
                        {                        
                            cloudLabel[ind] = 2;
                            cornerCloudSharp->push_back(lidarCloud->points[ind]);
                            cornerCloud->push_back(lidarCloud->points[ind]);
                        }
                        else if (largestPickedNum <= 20)
                        {                        
                            cloudLabel[ind] = 1; 
                            cornerCloud->push_back(lidarCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1; 

                        for (int l = 1; l <= 5; l++)
                        {
                            float diffX = lidarCloud->points[ind + l].x - lidarCloud->points[ind + l - 1].x;
                            float diffY = lidarCloud->points[ind + l].y - lidarCloud->points[ind + l - 1].y;
                            float diffZ = lidarCloud->points[ind + l].z - lidarCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = lidarCloud->points[ind + l].x - lidarCloud->points[ind + l + 1].x;
                            float diffY = lidarCloud->points[ind + l].y - lidarCloud->points[ind + l + 1].y;
                            float diffZ = lidarCloud->points[ind + l].z - lidarCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;

                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1; 
                        surfaceCloudFlat->push_back(lidarCloud->points[ind]);
                        smallestPickedNum++;
                        if (smallestPickedNum >= 4)
                        { 
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++)
                        { 
                            float diffX = lidarCloud->points[ind + l].x - lidarCloud->points[ind + l - 1].x;
                            float diffY = lidarCloud->points[ind + l].y - lidarCloud->points[ind + l - 1].y;
                            float diffZ = lidarCloud->points[ind + l].z - lidarCloud->points[ind + l - 1].z;

                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }


                            cloudNeighborPicked[ind + l] = 1;

                        }
                        
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = lidarCloud->points[ind + l].x - lidarCloud->points[ind + l + 1].x;
                            float diffY = lidarCloud->points[ind + l].y - lidarCloud->points[ind + l + 1].y;
                            float diffZ = lidarCloud->points[ind + l].z - lidarCloud->points[ind + l + 1].z;

                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfaceCloudTmp->push_back(lidarCloud->points[k]);
                    }
                }
            }
            TicToc tmp;
            
            pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());
            
            downSizeFilter.setInputCloud(surfaceCloudTmp);
            downSizeFilter.filter(*surfaceCloudScanDS);
            
            t_filter += tmp.toc();
            *surfaceCloud += *surfaceCloudScanDS;
            
        }
        // printf("sort q time %f \n", t_q_sort);
        // printf("seperate points time %f \n", t_pts.toc());
        // printf("filter points time %f \n", t_filter);
