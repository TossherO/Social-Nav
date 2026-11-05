    void markBadPoints(pcl::PointCloud<PointType>::Ptr cloudIn)
    {
        int cloudSize = cloudIn->size();
          for (int i = 5; i < cloudSize - 6; i++) 
          {
              
            float diffX = cloudIn->points[i + 1].x - cloudIn->points[i].x;
            float diffY = cloudIn->points[i + 1].y - cloudIn->points[i].y;
            float diffZ = cloudIn->points[i + 1].z - cloudIn->points[i].z;
            
            float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;
            // 0.2 horizontal angle accuracy, so dist > 0.1/(0.2/57.3) = 28.65 m
            if (diff > lidarMinRange) 
            {
                float depth1 = sqrt(cloudIn->points[i].x * cloudIn->points[i].x + 
                                cloudIn->points[i].y * cloudIn->points[i].y +
                                cloudIn->points[i].z * cloudIn->points[i].z);

                
                float depth2 = sqrt(cloudIn->points[i + 1].x * cloudIn->points[i + 1].x + 
                                cloudIn->points[i + 1].y * cloudIn->points[i + 1].y +
                                cloudIn->points[i + 1].z * cloudIn->points[i + 1].z);

               
                if (depth1 > depth2) {
                    diffX = cloudIn->points[i + 1].x - cloudIn->points[i].x * depth2 / depth1;
                    diffY = cloudIn->points[i + 1].y - cloudIn->points[i].y * depth2 / depth1;
                    diffZ = cloudIn->points[i + 1].z - cloudIn->points[i].z * depth2 / depth1;

                    
                    if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {
                        
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                    }
                } else {
                    diffX = cloudIn->points[i + 1].x * depth1 / depth2 - cloudIn->points[i].x;
                    diffY = cloudIn->points[i + 1].y * depth1 / depth2 - cloudIn->points[i].y;
                    diffZ = cloudIn->points[i + 1].z * depth1 / depth2 - cloudIn->points[i].z;

                    if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                    }
                }
            }

            float diffX2 = cloudIn->points[i].x - cloudIn->points[i - 1].x;
            float diffY2 = cloudIn->points[i].y - cloudIn->points[i - 1].y;
            float diffZ2 = cloudIn->points[i].z - cloudIn->points[i - 1].z;

            float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;
            float dis = cloudIn->points[i].x * cloudIn->points[i].x
                    + cloudIn->points[i].y * cloudIn->points[i].y
                    + cloudIn->points[i].z * cloudIn->points[i].z;
            if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) 
            {
                cloudNeighborPicked[i] = 1;
            }
        }
    }
