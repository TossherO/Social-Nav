#ifdef SOCIAL_NAV_NATIVE_ROLL
#include "roll_support.h"
#else
#include"utility.h"
#endif
#ifdef SOCIAL_NAV_LIO_CUDA
#include "cuda_matcher.h"
#endif



class LOAMmapping : public ParamServer
{
    private: 
        float transformTobeMapped[6];
        float transformGuess[6];
        std::vector<float, Eigen::aligned_allocator<float>> jacobianWorkspace, residualWorkspace;
        Eigen::Matrix<float, 6, 6> degeneracyProjection = Eigen::Matrix<float, 6, 6>::Identity();
#ifdef SOCIAL_NAV_LIO_CUDA
        std::unique_ptr<LioCuda> cudaCorner,cudaSurface;
        std::vector<LioPoint> cudaQueries;
        std::vector<LioMatch> cudaCornerMatches,cudaSurfaceMatches;
        uint64_t cudaMapRevision=UINT64_MAX;
        bool cudaUseCorners=false;
        bool cudaPlaneFit=false;
        void cuda_map(LioCuda& gpu,const pcl::PointCloud<PointType>& cloud) {
            if(cloud.size()>=(1u<<24))throw std::runtime_error("ROLL CUDA map exceeds exact point identity range");
            cudaQueries.resize(cloud.size());
            for(size_t i=0;i<cloud.size();++i){const auto& p=cloud[i];cudaQueries[i]={p.x,p.y,p.z,float(i)};}
            gpu.reset(cudaQueries.data(),cudaQueries.size());
        }
        void cuda_search(LioCuda& gpu,const pcl::PointCloud<PointType>& cloud,std::vector<LioMatch>& matches) {
            cudaQueries.resize(cloud.size());matches.resize(cloud.size());
            for(size_t i=0;i<cloud.size();++i){PointType p;pointAssociateToMap(&cloud[i],&p);cudaQueries[i]={p.x,p.y,p.z,0};}
            gpu.search(cudaQueries.data(),cudaQueries.size(),1.f,matches.data());
        }
#endif
    public: 
        struct IterationStats {
            int iteration, correspondences;
            bool degenerate;
            double min_eigen, residual;
            float raw_rotation, raw_translation;
            Eigen::Matrix<float, 6, 1> before, step;
        };
        // Detailed solver diagnostics are opt-in, like the run's other CSV outputs.
        bool traceEnabled = false;
        uint64_t cudaFallbacks=0;
#ifdef SOCIAL_NAV_LIO_CUDA
        void enableCuda(int device,float cell,bool corners,bool plane_fit){
            cudaUseCorners=corners;cudaCorner.reset(new LioCuda(device,.2f));cudaSurface.reset(new LioCuda(device,.2f));
            cudaPlaneFit=plane_fit;cudaSurface->surface_fitting(plane_fit);
            cudaCorner->search_resolution(cell);cudaSurface->search_resolution(cell);}
        LioCudaStats cudaStats()const {
            LioCudaStats s{};if(!cudaCorner)return s;auto a=cudaCorner->stats(),b=cudaSurface->stats();
            s.points=a.points+b.points;s.h2d_bytes=a.h2d_bytes+b.h2d_bytes;s.d2h_bytes=a.d2h_bytes+b.d2h_bytes;
            s.map_ms=a.map_ms+b.map_ms;s.match_ms=a.match_ms+b.match_ms;s.query_device_ms=a.query_device_ms+b.query_device_ms;return s;
        }
#endif
        void beginFrame(){cornerTime=surfTime=optTime=solveTime=0;iterCount=0;cudaFallbacks=0;ranMatch=false;
#ifdef SOCIAL_NAV_LIO_CUDA
            if(cudaCorner){cudaCorner->frame_begin();cudaSurface->frame_begin();}
#endif
        }
        std::vector<IterationStats, Eigen::aligned_allocator<IterationStats>> iterationTrace;
        Eigen::Affine3f affine_out;
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
        pcl::PointCloud<PointType>::Ptr lidarCloudCornerLastDS;
        pcl::PointCloud<PointType>::Ptr lidarCloudSurfLastDS;
        pcl::PointCloud<PointType>::Ptr lidarCloudSurfFromMapDS;
        pcl::PointCloud<PointType>::Ptr lidarCloudCornerFromMapDS;

        pcl::PointCloud<PointType>::Ptr lidarCloudOri;
        pcl::PointCloud<PointType>::Ptr coeffSel;

        std::vector<PointType> lidarCloudOriCornerVec; // corner point holder for parallel computation
        std::vector<PointType> coeffSelCornerVec;
        std::vector<uint8_t> lidarCloudOriCornerFlag;
        std::vector<PointType> lidarCloudOriSurfVec; // surf point holder for parallel computation
        std::vector<PointType> coeffSelSurfVec;
        std::vector<uint8_t> lidarCloudOriSurfFlag;
        vector<double> mapRegistrationError;

        
        int iterCount = 0;
        int lidarCloudCornerLastDSNum = 0;
        int lidarCloudSurfLastDSNum = 0;

        int edgePointCorrNum = 0;
        int surfPointCorrNum = 0;

        double inlier_ratio = 0;
        double inlier_ratio2 = 0;
        float cornerTime = 0, surfTime = 0, optTime = 0, solveTime=0;
        bool ranMatch=false;
        double regiError;

        double minEigen = 1e+6;
        Eigen::Matrix<float, 6, 1> firstEigenvalues = Eigen::Matrix<float, 6, 1>::Zero();
        Eigen::Matrix<float, 6, 1> weakestDirection = Eigen::Matrix<float, 6, 1>::Zero();

        bool isDegenerate = false;

        explicit LOAMmapping(const ParamServer& params) : ParamServer(params)
        {
            lidarCloudOri.reset(new pcl::PointCloud<PointType>());
            coeffSel.reset(new pcl::PointCloud<PointType>());
        }

        LOAMmapping(pcl::PointCloud<PointType>::Ptr lidarCloudCornerLastDSnew,pcl::PointCloud<PointType>::Ptr lidarCloudSurfLastDSnew,
            pcl::PointCloud<PointType>::Ptr lidarCloudCornerFromMapDSnew, pcl::PointCloud<PointType>::Ptr lidarCloudSurfFromMapDSnew,
            const Eigen::Affine3f affine_guess_new,
#ifdef SOCIAL_NAV_NATIVE_ROLL
            const ParamServer& params,
#else
            const ParamServer& params = ParamServer(),
#endif
            pcl::KdTreeFLANN<PointType>::Ptr cornerTree = {},
            pcl::KdTreeFLANN<PointType>::Ptr surfaceTree = {})
            : LOAMmapping(params)
        {
            reset(lidarCloudCornerLastDSnew, lidarCloudSurfLastDSnew, lidarCloudCornerFromMapDSnew,
                  lidarCloudSurfFromMapDSnew, affine_guess_new, cornerTree, surfaceTree);
        }

        void reset(pcl::PointCloud<PointType>::Ptr lidarCloudCornerLastDSnew,
                   pcl::PointCloud<PointType>::Ptr lidarCloudSurfLastDSnew,
                   pcl::PointCloud<PointType>::Ptr lidarCloudCornerFromMapDSnew,
                   pcl::PointCloud<PointType>::Ptr lidarCloudSurfFromMapDSnew,
                   const Eigen::Affine3f affine_guess_new,
                   pcl::KdTreeFLANN<PointType>::Ptr cornerTree = {},
                   pcl::KdTreeFLANN<PointType>::Ptr surfaceTree = {},uint64_t geometryRevision=0)
        {
            // Matching only reads these clouds. The caller holds the map lock for
            // this match and invalidates the shared trees whenever geometry changes.
            lidarCloudCornerFromMapDS = lidarCloudCornerFromMapDSnew;
            lidarCloudSurfFromMapDS = lidarCloudSurfFromMapDSnew;
            lidarCloudCornerLastDS = lidarCloudCornerLastDSnew;
            lidarCloudSurfLastDS = lidarCloudSurfLastDSnew;
            kdtreeCornerFromMap = cornerTree;
            kdtreeSurfFromMap = surfaceTree;
#ifdef SOCIAL_NAV_LIO_CUDA
            if(cudaCorner){
                if(!localizationMode||cudaMapRevision!=geometryRevision){
                    if(cudaUseCorners)cuda_map(*cudaCorner,*lidarCloudCornerFromMapDS);
                    cuda_map(*cudaSurface,*lidarCloudSurfFromMapDS);cudaMapRevision=geometryRevision;}}
#endif
            cudaFallbacks=0;
            if (!kdtreeCornerFromMap) {
                kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
                kdtreeCornerFromMap->setInputCloud(lidarCloudCornerFromMapDS);
            }
            if (!kdtreeSurfFromMap) {
                kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());
                kdtreeSurfFromMap->setInputCloud(lidarCloudSurfFromMapDS);
            }

            lidarCloudCornerLastDSNum = lidarCloudCornerLastDS->points.size();
            lidarCloudSurfLastDSNum = lidarCloudSurfLastDS->points.size();
            affine_out = affine_guess_new;

            Affine3f2Trans(affine_out, transformTobeMapped);
            Affine3f2Trans(affine_out, transformGuess);

            const size_t corners = lidarCloudCornerLastDS->size(), surfaces = lidarCloudSurfLastDS->size();
            lidarCloudOriCornerVec.resize(corners);
            coeffSelCornerVec.resize(corners);
            lidarCloudOriCornerFlag.resize(corners);
            lidarCloudOriSurfVec.resize(surfaces);
            coeffSelSurfVec.resize(surfaces);
            lidarCloudOriSurfFlag.resize(surfaces);
            lidarCloudOri->clear(); coeffSel->clear(); mapRegistrationError.clear();
            lidarCloudOri->reserve(corners + surfaces); coeffSel->reserve(corners + surfaces);
            mapRegistrationError.reserve(corners + surfaces);
            jacobianWorkspace.resize((corners + surfaces)*6);
            residualWorkspace.resize(corners + surfaces);
            edgePointCorrNum = surfPointCorrNum = iterCount = 0;
            inlier_ratio = inlier_ratio2 = cornerTime = surfTime = optTime = solveTime = 0;
            regiError = -1; minEigen = 1e6; isDegenerate = false;
            degeneracyProjection.setIdentity();
            iterationTrace.clear();
            firstEigenvalues.setZero(); weakestDirection.setZero();

            std::fill(lidarCloudOriCornerFlag.begin(), lidarCloudOriCornerFlag.end(), false);
            std::fill(lidarCloudOriSurfFlag.begin(), lidarCloudOriSurfFlag.end(), false);
            
        }

        void match( )
        {
            ranMatch=true;
            iterCount = 0;
            for (; iterCount < optIteration; iterCount++)
            {
                lidarCloudOri->clear();
                coeffSel->clear();

                if(edgeEnabled == true)
                {
                    TicToc corner;
                    cornerOptimization(iterCount);
                    cornerTime += corner.toc();
                }


                TicToc surf;
                surfOptimization(iterCount);
                surfTime += surf.toc();
                TicToc opt;
                combineOptimizationCoeffs();
                optTime += opt.toc();
                inlier_ratio2 = (double)(surfPointCorrNum + edgePointCorrNum)/( lidarCloudSurfLastDSNum+ lidarCloudCornerLastDSNum);
                // surf only
                // inlier_ratio2 = (double)surfPointCorrNum/ lidarCloudSurfLastDSNum;

                // it is actually for map extension, otherwise cv error happens
                // 0.2 or higher would seriously deteriarate the performance
                if (inlier_ratio2 < 0.1 ) break; 
                TicToc solve;
                bool converged=LMOptimization(iterCount);
                solveTime+=solve.toc();
                if (converged)
                {
                    // cout<<"converged"<<endl;
                    break;              
                }
            }
            // The final LM step may terminate the loop before the next feature pass.
            affine_out = trans2Affine3f(transformTobeMapped);
            // cout<<cornerTime<<" "<<surfTime<<" edge corr: "<<edgePointCorrNum <<" surf corr: "<<surfPointCorrNum<<endl; 
            // lidarCloudSelNum = errorSize
            
            if (mapRegistrationError.empty()) 
            {
                inlier_ratio = 0;
                minEigen = 0;
                regiError = 0;
                return;
            }
            regiError = accumulate(mapRegistrationError.begin(),mapRegistrationError.end(),0.0);
            double errorSize = mapRegistrationError.size();
            regiError /= errorSize;                        
            int inlierCnt = 0;
            for (int i = 0; i < errorSize; i++)
            {
                // regiError2 += mapRegistrationError[i]*mapRegistrationError[i];
                if (mapRegistrationError[i] < inlierThreshold) inlierCnt++; // <0.1 means "properly matched"
            }
            // regiError2 = sqrt(regiError2/errorSize);
            inlier_ratio = inlierCnt/ errorSize;
            // ROS_INFO_STREAM(" mapping error: "<<regiError);
        }


    void cornerOptimization(int iterCount)
    {
        affine_out = trans2Affine3f(transformTobeMapped);
#ifdef SOCIAL_NAV_LIO_CUDA
        if(cudaCorner&&cudaUseCorners)cuda_search(*cudaCorner,*lidarCloudCornerLastDS,cudaCornerMatches);
#endif

        // FLANN reads an immutable index. All query, fit and output scratch is
        // thread-local; each point writes its own slot, collected in scan order.
        uint64_t fallback_count=0;
        #pragma omp parallel num_threads(matchingThreads) if(lidarCloudCornerLastDSNum >= 256) reduction(+:fallback_count)
        {
        std::vector<int> pointSearchInd(5);
        std::vector<float> pointSearchSqDis(5);
        #pragma omp for schedule(static)
        for (int i = 0; i < lidarCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;

            pointOri = lidarCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            int found;
#ifdef SOCIAL_NAV_LIO_CUDA
            if(cudaCorner && cudaUseCorners && !(cudaCornerMatches[i].flags&1)){
                const auto& m=cudaCornerMatches[i];found=m.count;
                for(int j=0;j<found;++j){pointSearchInd[j]=int(m.neighbors[j].intensity);pointSearchSqDis[j]=m.distance[j];}
            }else
#endif
            {found=kdtreeCornerFromMap->nearestKSearch(pointSel,5,pointSearchInd,pointSearchSqDis);
#ifdef SOCIAL_NAV_LIO_CUDA
                if(cudaCorner&&cudaUseCorners){
                    fallback_count++;
                }
#endif
            }
            if(found<5)continue;


            Eigen::Matrix3f matA1 = Eigen::Matrix3f::Identity();
                    
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += lidarCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += lidarCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += lidarCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = lidarCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = lidarCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = lidarCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1(0, 0) = a11; matA1(0, 1) = a12; matA1(0, 2) = a13;
                matA1(1, 0) = a12; matA1(1, 1) = a22; matA1(1, 2) = a23;
                matA1(2, 0) = a13; matA1(2, 1) = a23; matA1(2, 2) = a33;

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(matA1);
                Eigen::Vector3f matD1 = solver.eigenvalues().real(); // from small to big
                Eigen::Matrix3f matV1 = solver.eigenvectors().real();
                
                // 注意：cv::eigen得到的matV每行都是特征向量，matD从大到小
                // 对于eigen, 则是每列，且从小到大
                if (matD1(2, 0) > 3 * matD1(1, 0)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // eigenvector corresponding to the biggest eigenvalue
                    float x1 = cx + 0.1 * matV1(0, 2);
                    float y1 = cy + 0.1 * matV1(1, 2);
                    float z1 = cz + 0.1 * matV1(2, 2);
                    float x2 = cx - 0.1 * matV1(0, 2);
                    float y2 = cy - 0.1 * matV1(1, 2);
                    float z2 = cz - 0.1 * matV1(2, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) 
                    {
                        lidarCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        lidarCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
        }
        cudaFallbacks+=fallback_count;
    }

    void surfOptimization(int iterCount)
    {
        affine_out = trans2Affine3f(transformTobeMapped);
#ifdef SOCIAL_NAV_LIO_CUDA
        if(cudaSurface)cuda_search(*cudaSurface,*lidarCloudSurfLastDS,cudaSurfaceMatches);
#endif

        uint64_t fallback_count=0;
        #pragma omp parallel num_threads(matchingThreads) if(lidarCloudSurfLastDSNum >= 256) reduction(+:fallback_count)
        {
        std::vector<int> pointSearchInd(5);
        std::vector<float> pointSearchSqDis(5);
        #pragma omp for schedule(static)
        for (int i = 0; i < lidarCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;

            pointOri = lidarCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
#ifdef SOCIAL_NAV_LIO_CUDA
            if(cudaSurface&&cudaPlaneFit&&(cudaSurfaceMatches[i].flags&2))fallback_count++;
            if(cudaSurface&&cudaPlaneFit&&!(cudaSurfaceMatches[i].flags&3)){
                const auto& m=cudaSurfaceMatches[i];
                if(m.selected){coeff.x=m.normal[0];coeff.y=m.normal[1];coeff.z=m.normal[2];coeff.intensity=m.normal[3];
                    lidarCloudOriSurfVec[i]=pointOri;coeffSelSurfVec[i]=coeff;lidarCloudOriSurfFlag[i]=true;}
                continue;
            }
#endif
            int found;
#ifdef SOCIAL_NAV_LIO_CUDA
            if(cudaSurface && !(cudaSurfaceMatches[i].flags&1)){
                const auto& m=cudaSurfaceMatches[i];found=m.count;
                for(int j=0;j<found;++j){pointSearchInd[j]=int(m.neighbors[j].intensity);pointSearchSqDis[j]=m.distance[j];}
            }else
#endif
            {found=kdtreeSurfFromMap->nearestKSearch(pointSel,5,pointSearchInd,pointSearchSqDis);
#ifdef SOCIAL_NAV_LIO_CUDA
                if(cudaSurface){
                    fallback_count++;
                }
#endif
            }
            if(found<5)continue;

            
            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) 
                {
                    matA0(j, 0) = lidarCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = lidarCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = lidarCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // why Ax = B, means x is the unit normal vector of the plane?
                // 
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * lidarCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * lidarCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * lidarCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;
                    if (s > 0.1) {
                        lidarCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        lidarCloudOriSurfFlag[i] = true;   
                    }
                }

            }
        }
        }
        cudaFallbacks+=fallback_count;
    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        edgePointCorrNum = 0;
        surfPointCorrNum = 0;
        for (int i = 0; i < lidarCloudCornerLastDSNum; ++i){
            if (lidarCloudOriCornerFlag[i] == true){
                lidarCloudOri->push_back(lidarCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
                edgePointCorrNum++;
            }
        }
        // combine surf coeffs
        for (int i = 0; i < lidarCloudSurfLastDSNum; ++i){
            if (lidarCloudOriSurfFlag[i] == true){
                lidarCloudOri->push_back(lidarCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
                surfPointCorrNum++;
            }
        }
        // reset flag for next iteration
        std::fill(lidarCloudOriCornerFlag.begin(), lidarCloudOriCornerFlag.end(), false);
        std::fill(lidarCloudOriSurfFlag.begin(), lidarCloudOriSurfFlag.end(), false);


    }

    bool LMOptimization(int iterCount)
    {
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int lidarCloudSelNum = lidarCloudOri->size();

        // Contiguous maps retain the original reduction layout without an
        // allocation or zero-fill on every LM iteration. Every entry is set below.
        Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> matA(jacobianWorkspace.data(), lidarCloudSelNum, 6);
        Eigen::Matrix<float,6,6> matAtA = Eigen::Matrix<float,6,6>::Zero();
        Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> matB(residualWorkspace.data(), lidarCloudSelNum, 1);
        Eigen::Matrix<float,6,1> matAtB = Eigen::Matrix<float,6,1>::Zero();
        Eigen::Matrix<float,6,1> matX = Eigen::Matrix<float,6,1>::Zero();

        PointType pointOri, coeff;
        mapRegistrationError.clear(); // only get the last iteration error
        for (int i = 0; i < lidarCloudSelNum; i++) {
            mapRegistrationError.push_back(fabs(coeffSel->points[i].intensity));
            // lidar -> camera
            pointOri.x = lidarCloudOri->points[i].y;
            pointOri.y = lidarCloudOri->points[i].z;
            pointOri.z = lidarCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // camera -> lidar
            matA(i, 0) = arz;
            matA(i, 1) = arx;
            matA(i, 2) = ary;
            matA(i, 3) = coeff.z;
            matA(i, 4) = coeff.x;
            matA(i, 5) = coeff.y;
            matB(i, 0) = -coeff.intensity;
        }
        
        matAtA = matA.transpose() * matA;
        matAtB = matA.transpose() * matB;
        // cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
        matX = matAtA.colPivHouseholderQr().solve(matAtB);
        const float rawRotation = matX.head<3>().norm(), rawTranslation = matX.tail<3>().norm();

        if (iterCount == 0) 
        {
            // cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            // cv::Mat matVf(6, 6, CV_32F, cv::Scalar::all(0));
            // cv::Mat matVu(6, 6, CV_32F, cv::Scalar::all(0));

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> solver(matAtA);
            Eigen::Matrix<float,6,1> matE = solver.eigenvalues().real();
            Eigen::Matrix<float,6,6> matVf = solver.eigenvectors().real();

            // 注意：cv::eigen得到的matV每行都是特征向量，matD从大到小
            // 对于eigen, 则是每列，且从小到大

            // cv::eigen(matAtA, matE, matVf);
            // matVf.copyTo(matVu);
            Eigen::Matrix<float,6,6> matVu = matVf; // checked it is a shallow copy

            isDegenerate = false;
            
            firstEigenvalues = matE; weakestDirection = matVf.col(0);
            Eigen::Matrix<float,6,1> eigenThre = degeneracyEigenThreshold*Eigen::Matrix<float,6,1>::Ones();
            minEigen = matE(0);
            for (int i = 0; i <= 5; i++) { // from small to big
                if (matE(i) < eigenThre(i)) 
                {
                    for (int j = 0; j < 6; j++) 
                    {
                        matVu(j, i) = 0; // zero out the underconstrained DOF colomn, which will not be updated on transformTobeMapped
                    }
                    isDegenerate = true;
                }
                else 
                { 
                    break;
                }
            }
            // Keep this scan's observable subspace for every LM iteration. A
            // local zero matrix would erase the second step and fake convergence.
            degeneracyProjection = matVf * matVu.transpose();
        }
        
        if (isDegenerate)
        {
            Eigen::Matrix<float,6,1> matX2 = matX;
            matX = degeneracyProjection * matX2;
        }

        if (traceEnabled) {
            IterationStats stats;
            stats.iteration = iterCount; stats.correspondences = lidarCloudSelNum;
            stats.degenerate = isDegenerate; stats.min_eigen = minEigen;
            stats.residual = std::accumulate(mapRegistrationError.begin(), mapRegistrationError.end(), 0.0)
                             / std::max(1, lidarCloudSelNum);
            stats.raw_rotation = rawRotation; stats.raw_translation = rawTranslation;
            stats.before = Eigen::Map<Eigen::Matrix<float, 6, 1>>(transformTobeMapped);
            stats.step = matX;
            iterationTrace.push_back(stats);
        }
        
        transformTobeMapped[0] += matX(0, 0);
        transformTobeMapped[1] += matX(1, 0);
        transformTobeMapped[2] += matX(2, 0);
        transformTobeMapped[3] += matX(3, 0);
        transformTobeMapped[4] += matX(4, 0);
        transformTobeMapped[5] += matX(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX(0, 0)), 2) +
                            pow(pcl::rad2deg(matX(1, 0)), 2) +
                            pow(pcl::rad2deg(matX(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX(3, 0) * 100, 2) +
                            pow(matX(4, 0) * 100, 2) +
                            pow(matX(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }



    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = affine_out(0,0) * pi->x + affine_out(0,1) * pi->y + affine_out(0,2) * pi->z + affine_out(0,3);
        po->y = affine_out(1,0) * pi->x + affine_out(1,1) * pi->y + affine_out(1,2) * pi->z + affine_out(1,3);
        po->z = affine_out(2,0) * pi->x + affine_out(2,1) * pi->y + affine_out(2,2) * pi->z + affine_out(2,3);
        po->intensity = pi->intensity;
    }

    void getTransformation(float transformTobeMappedI[6])
    {
        for (int i = 0; i < 6; i++) transformTobeMappedI[i] = transformTobeMapped[i];
    }

};
