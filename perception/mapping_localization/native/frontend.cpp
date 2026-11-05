// FAST-LIO numerical routines stay in FrontendMath.inc; no ROS linkage here.
#include "frontend.h"
#include <omp.h>
#include <queue>
#include <mutex>
#include <limits>
#include <cstring>
#include <yaml-cpp/yaml.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include "IMU_Processing.hpp"
#include "MotionInitializer.h"
#include <ikd-Tree/ikd_Tree.h>
#ifdef SOCIAL_NAV_LIO_CUDA
#include "cuda_matcher.h"
#endif

#define pairPQ priority_queue<pair<double,int>,vector<pair<double,int>>,less<pair<double,int>>>

class Frontend {
    bool local_state_en=true, debugMode=false, info_filtering=false, extrinsic_est_en=false;
    int local_min_points=20, feats_down_size=0, effct_feat_num=0, NUM_MAX_ITERATIONS=3;
    int matching_threads=1;
    int kdtree_delete_counter=0, add_point_size=0;
    double local_min_ratio=.1, local_max_error=.2, local_max_range=50;
    double res_mean_last=INFINITY, total_residual=0, match_time=0, solve_time=0;
    double cube_len=200, filter_size_map_min=.2, kdtree_delete_time=0, kdtree_incremental_time=0;
    float DET_RANGE=50;
    const float MOV_THRESHOLD=1.5;
    bool Localmap_Initialized=false, flg_EKF_inited=false, initialized=false;
    bool adaptive_initialization=true;
    double first_lidar_time=0, last_stamp=-1;
    bool point_selected_surf[100000];
    float res_last[100000];
    std::mutex getMap;
    BoxPointType LocalMap_Points;
    vector<BoxPointType> cub_needrm;
    vector<PointVector> Nearest_Points;
    KD_TREE ikdtree;
    V3F XAxisPoint_body{LIDAR_SP_LEN,0,0}, XAxisPoint_world{LIDAR_SP_LEN,0,0};
    state_ikfom state_point;
    V3D extrinsic_t=V3D::Zero();M3D extrinsic_r=M3D::Identity();
    vect3 pos_lid;
    PointCloudXYZI::Ptr feats_undistort{new PointCloudXYZI}, feats_down_body{new PointCloudXYZI};
    PointCloudXYZI::Ptr feats_down_world{new PointCloudXYZI}, normvec{new PointCloudXYZI(100000,1)};
    PointCloudXYZI::Ptr laserCloudOri{new PointCloudXYZI(100000,1)}, corr_normvect{new PointCloudXYZI(100000,1)};
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    ImuProcess imu;
    social_nav_init::MotionInitializer initializer;
    esekfom::esekf<state_ikfom,12,input_ikfom> kf;
    SnLioStages stages{};
#ifdef SOCIAL_NAV_LIO_CUDA
    std::unique_ptr<LioCuda> cuda_matcher;
    std::vector<LioPoint> cuda_body,cuda_world;
    std::vector<LioMatch> cuda_matches,cuda_fixes;
    std::vector<int> cuda_fix_ids;
    std::vector<uint8_t> cuda_selected;
    bool cuda_plane_fit=false;
    bool cuda_linearization=false;
    std::unique_ptr<LioCuda> cuda_deskewer;
    std::vector<LioTimedPoint> cuda_timed;
    std::vector<LioImuPose> cuda_imu;
    std::vector<LioPoint> cuda_normals;
    uint64_t nn_fallbacks=0,plane_fallbacks=0;
    static LioPoint gpu_point(const PointType& p){return {p.x,p.y,p.z,p.intensity};}
    static LioBox gpu_box(const BoxPointType& b){LioBox out{};
        for(int j=0;j<3;++j){out.low[j]=b.vertex_min[j];out.high[j]=b.vertex_max[j];}return out;}
    void cuda_deskew(PointCloudXYZI& cloud,const vector<Pose6D>& poses,const state_ikfom& state) {
        if(poses.size()<2||cloud.empty())return;
        cuda_timed.resize(cloud.size());cuda_imu.resize(poses.size());
        for(size_t i=0;i<cloud.size();++i){const auto& p=cloud[i];cuda_timed[i]={p.x,p.y,p.z,p.intensity,p.curvature};}
        for(size_t i=0;i<poses.size();++i){const auto& p=poses[i];auto& q=cuda_imu[i];q.offset=p.offset_time;
            for(int j=0;j<9;++j)q.rotation[j]=p.rot[j];
            for(int j=0;j<3;++j){q.velocity[j]=p.vel[j];q.position[j]=p.pos[j];q.acceleration[j]=p.acc[j];q.angular_velocity[j]=p.gyr[j];}}
        LioDeskewState s{};auto extrinsic=state.offset_R_L_I.toRotationMatrix();
        auto inverse=state.rot.conjugate().toRotationMatrix(),inverse_ext=state.offset_R_L_I.conjugate().toRotationMatrix();
        for(int i=0;i<3;++i){s.translation[i]=state.offset_T_L_I[i];s.position[i]=state.pos[i];
            for(int j=0;j<3;++j){s.extrinsic[3*i+j]=extrinsic(i,j);s.inverse_rotation[3*i+j]=inverse(i,j);s.inverse_extrinsic[3*i+j]=inverse_ext(i,j);}}
        cuda_deskewer->deskew(cuda_timed.data(),cuda_timed.size(),cuda_imu.data(),cuda_imu.size(),s);
        for(size_t i=0;i<cloud.size();++i){cloud[i].x=cuda_timed[i].x;cloud[i].y=cuda_timed[i].y;cloud[i].z=cuda_timed[i].z;}
    }
    void cuda_linearize(state_ikfom& s,esekfom::dyn_share_datastruct<double>& data) {
        double state[21];auto rotation=s.offset_R_L_I.toRotationMatrix(),inverse=s.rot.conjugate().toRotationMatrix();
        for(int i=0;i<3;++i){state[9+i]=s.offset_T_L_I[i];for(int j=0;j<3;++j){state[3*i+j]=rotation(i,j);state[12+3*i+j]=inverse(i,j);}}
        const LioPoint* normals=nullptr;
        if(!cuda_plane_fit){cuda_normals.resize(feats_down_size);
            for(int i=0;i<feats_down_size;++i){const auto& n=normvec->points[i];cuda_normals[i]={n.x,n.y,n.z,n.intensity};}normals=cuda_normals.data();}
        cuda_matcher->linearize(normals,feats_down_size,state,extrinsic_est_en,data.normal_matrix.data(),data.normal_rhs.data());
        data.normal_count=effct_feat_num;
    }
    void cuda_match(state_ikfom& s,esekfom::dyn_share_datastruct<double>& data) {
        const int n=feats_down_size;cuda_world.resize(n);cuda_matches.resize(n);cuda_selected.resize(n);
        for(int i=0;i<n;++i){const auto& p=feats_down_body->points[i];
            V3D body(p.x,p.y,p.z),world(s.rot*(s.offset_R_L_I*body+s.offset_T_L_I)+s.pos);
            auto& w=feats_down_world->points[i];w.x=world.x();w.y=world.y();w.z=world.z();w.intensity=p.intensity;
            cuda_world[i]=gpu_point(w);}
        cuda_matcher->match(cuda_world.data(),n,data.converge,cuda_matches.data());
        cuda_fixes.clear();cuda_fix_ids.clear();std::vector<float> distances(NUM_MATCH_POINTS);
        for(int i=0;i<n;++i){auto& m=cuda_matches[i];auto& near=Nearest_Points[i];
            const auto& w=feats_down_world->points[i];const auto& b=feats_down_body->points[i];
            if(data.converge){near.resize(m.count);
                for(int j=0;j<m.count;++j){auto& p=near[j];p=PointType{};p.x=m.neighbors[j].x;p.y=m.neighbors[j].y;p.z=m.neighbors[j].z;p.intensity=m.neighbors[j].intensity;}}
            if(m.flags&1){++nn_fallbacks;ikdtree.Nearest_Search(w,NUM_MATCH_POINTS,near,distances);
                m.count=near.size();m.selected=m.count==NUM_MATCH_POINTS&&distances[NUM_MATCH_POINTS-1]<=5;
                for(int j=0;j<m.count;++j){m.neighbors[j]=gpu_point(near[j]);m.distance[j]=distances[j];}}
            if(m.flags){++plane_fallbacks;
                // Plane-only requests were selected before the device fit.
                bool eligible=(m.flags&1)?bool(m.selected):true;m.selected=0;
                VF(4) plane;
                if(eligible&&esti_plane(plane,near,.1f)){
                    float d=plane(0)*w.x+plane(1)*w.y+plane(2)*w.z+plane(3);
                    float score=1-.9*fabs(d)/sqrt(V3D(b.x,b.y,b.z).norm());
                    if(score>.9){for(int j=0;j<3;++j)m.normal[j]=plane(j);m.normal[3]=d;m.selected=1;}}
                bool patch_neighbors=m.flags&1;m.flags=0;
                if(cuda_plane_fit||patch_neighbors){cuda_fix_ids.push_back(i);cuda_fixes.push_back(m);}}
            point_selected_surf[i]=m.selected;
            cuda_selected[i]=m.selected;
            if(m.selected){auto& normal=normvec->points[i];normal.x=m.normal[0];normal.y=m.normal[1];normal.z=m.normal[2];normal.intensity=m.normal[3];res_last[i]=fabs(m.normal[3]);}
        }
        cuda_matcher->correct(cuda_fix_ids.data(),cuda_fixes.data(),cuda_fix_ids.size());
        if(!cuda_plane_fit)cuda_matcher->select(cuda_selected.data(),n);
    }
#endif
    #include "FrontendMath.inc"

    template<class T> static T option(const YAML::Node& node,const char* name,T fallback) {
        return node[name] ? node[name].as<T>() : fallback;
    }
public:
    explicit Frontend(const char* path) {
        auto c=YAML::LoadFile(path), m=c["mapping"], o=c["odometry"], ini=c["initialization"];
        if (option(m,"load_pcd",false) || option(c,"initialize_with_gt",false))
            throw std::invalid_argument("Native frontend takes continuous local motion; fixed-map alignment belongs to ROLL");
        local_state_en=option(o,"publish_state",false);
        local_min_points=option(o,"min_effective_points",20);
        local_min_ratio=option(o,"min_inlier_ratio",.1);
        local_max_error=option(o,"max_registration_error_m",.2);
        local_max_range=option(o,"max_range",50.);
        extrinsic_est_en=option(m,"extrinsic_est_en",true);
        info_filtering=option(c,"info_filtering",false);
        NUM_MAX_ITERATIONS=option(c,"max_iteration",4);
        matching_threads=option(c,"matching_threads",1);
        if(matching_threads<1)throw std::invalid_argument("matching_threads must be positive");
        cube_len=option(c,"cube_side_length",200.);
        DET_RANGE=option(m,"det_range",300.);
        filter_size_map_min=option(c,"filter_size_map",.5);
        double leaf=option(c,"filter_size_surf",.5);
        if (leaf<=0 || cube_len<=0 || filter_size_map_min<=0 || local_min_points<1)
            throw std::invalid_argument("Invalid native FAST-LIO geometry configuration");
        auto backend=option<std::string>(c,"matching_backend","cpu");
        auto deskew_backend=option<std::string>(c,"deskew_backend","cpu"),linearization=option<std::string>(c,"linearization_backend","cpu");
        if(backend!="cpu"&&backend!="cuda")throw std::invalid_argument("matching_backend must be cpu or cuda");
        if((deskew_backend!="cpu"&&deskew_backend!="cuda")||(linearization!="cpu"&&linearization!="cuda"))
            throw std::invalid_argument("deskew_backend and linearization_backend must be cpu or cuda");
        if(linearization=="cuda"&&backend!="cuda")throw std::invalid_argument("CUDA linearization requires CUDA matching");
        kf.compact_update=option(c,"filter_compact_solve",false);
#ifdef SOCIAL_NAV_LIO_CUDA
        cuda_linearization=linearization=="cuda";
        if(deskew_backend=="cuda"){
            cuda_deskewer.reset(new LioCuda(option(c,"matching_cuda_device",0),filter_size_map_min));
            imu.deskewer=[this](PointCloudXYZI& cloud,const vector<Pose6D>& poses,const state_ikfom& state){cuda_deskew(cloud,poses,state);};}
#else
        if(deskew_backend=="cuda")throw std::invalid_argument("CUDA deskew requires SOCIAL_NAV_BUILD_LIO_CUDA=ON");
#endif
        if(backend=="cuda"){
#ifdef SOCIAL_NAV_LIO_CUDA
            if(info_filtering)throw std::invalid_argument("CUDA matching does not support info_filtering");
            cuda_plane_fit=option(c,"matching_cuda_plane_fit",false);
            cuda_matcher.reset(new LioCuda(option(c,"matching_cuda_device",0),filter_size_map_min,cuda_plane_fit));
            ikdtree.point_inserted=[this](const PointType& p,const BoxPointType* b){LioBox box{};if(b)box=gpu_box(*b);cuda_matcher->insert(gpu_point(p),b?&box:nullptr);};
            ikdtree.box_erased=[this](const BoxPointType& b){cuda_matcher->erase(gpu_box(b));};
#else
            throw std::invalid_argument("CUDA matching requires a build with SOCIAL_NAV_BUILD_LIO_CUDA=ON");
#endif
        }
        downSizeFilterSurf.setLeafSize(leaf,leaf,leaf);
        memset(point_selected_surf,true,sizeof(point_selected_surf));
        memset(res_last,-1000,sizeof(res_last));
        auto t=m["extrinsic_T"].as<vector<double>>(), r=m["extrinsic_R"].as<vector<double>>();
        if(t.size()!=3 || r.size()!=9)throw std::invalid_argument("Invalid LiDAR-IMU extrinsics");
        V3D T(t[0],t[1],t[2]); M3D R; R<<MAT_FROM_ARRAY(r);
        extrinsic_t=T;extrinsic_r=R;
        imu.set_extrinsic(T,R);initializer.setExtrinsic(R,T);
        imu.need_map_erected=option(c,"need_map_erected",true);
        auto cov=[&](const char* name,double fallback){return V3D::Constant(option(m,name,fallback));};
        imu.set_acc_cov(cov("acc_cov",.1));imu.set_gyr_cov(cov("gyr_cov",.1));
        imu.set_acc_bias_cov(cov("b_acc_cov",.0001));imu.set_gyr_bias_cov(cov("b_gyr_cov",.0001));
        adaptive_initialization=option(ini,"enabled",false);
        auto& opts=initializer.options;
        opts.min_duration=option(ini,"min_duration",1.);opts.max_duration=option(ini,"max_duration",6.);
        opts.min_scans=option(ini,"min_scans",10);opts.voxel=option(ini,"voxel",.3);
        opts.acceleration_scale=option(ini,"acceleration_scale",1.);
        opts.max_residual=option(ini,"max_residual",.06);opts.max_acc_bias=option(ini,"max_acc_bias",1.5);
        opts.max_tilt_sensitivity=option(ini,"max_tilt_sensitivity",.01);
        opts.stable_duration=option(ini,"stable_duration",.3);
        imu.fixed_acceleration_scale=adaptive_initialization;imu.acceleration_scale=opts.acceleration_scale;
        double eps[23];std::fill(eps,eps+23,.001);
        kf.init_dyn_share(get_f,df_dx,df_dw,[this](state_ikfom& s,esekfom::dyn_share_datastruct<double>& d){h_share_model(s,d);},NUM_MAX_ITERATIONS,eps);
    }

    int step(const double* samples,size_t count,const float* cloud,size_t n,double begin,double end,SnLioResult& out) {
        const double start=omp_get_wtime();
        stages={};imu.prepare_ms=imu.predict_ms=imu.deskew_ms=0;solve_time=0;
#ifdef SOCIAL_NAV_LIO_CUDA
        if(cuda_matcher){cuda_matcher->frame_begin();nn_fallbacks=plane_fallbacks=0;}
        if(cuda_deskewer)cuda_deskewer->frame_begin();
#endif
        if(!std::isfinite(begin)||!std::isfinite(end)||end<begin||begin<=last_stamp||n>1000000)
            throw std::invalid_argument("Invalid or reversed FAST-LIO scan time/size");
        last_stamp=begin;out={};out.stamp=end;out.residual=INFINITY;
        MeasureGroup meas;meas.lidar_beg_time=begin;
        meas.lidar.reset(new PointCloudXYZI);meas.lidar->reserve(n);
        for(size_t i=0;i<n;++i){PointType p{};p.x=cloud[5*i];p.y=cloud[5*i+1];p.z=cloud[5*i+2];
            p.intensity=cloud[5*i+3];p.curvature=cloud[5*i+4];meas.lidar->push_back(p);}
        double prev=-INFINITY;
        for(size_t i=0;i<count;++i){const auto* v=samples+7*i;
            if(v[0]<=prev||v[0]>end+1e-7)throw std::invalid_argument("Noncausal or unordered IMU batch");prev=v[0];
            auto msg=std::make_shared<SocialNavImu>();msg->header.stamp.seconds=v[0];
            msg->linear_acceleration={v[1],v[2],v[3]};msg->angular_velocity={v[4],v[5],v[6]};meas.imu.push_back(msg);}
        match_time=kdtree_incremental_time=0;effct_feat_num=feats_down_size=0;res_mean_last=INFINITY;
        if(adaptive_initialization&&!initialized){
            if(initializer.process(meas.lidar,begin,meas.imu,*feats_undistort)){
                auto s=kf.get_x();s.rot=initializer.rotation_end;s.pos=initializer.position_end;
                s.offset_R_L_I=extrinsic_r;s.offset_T_L_I=extrinsic_t;
                s.vel=initializer.result.velocity;s.bg=initializer.result.gyro_bias;s.ba=initializer.result.acc_bias;
                s.grav=S2(initializer.result.gravity);imu.initializeMotion(meas,s,kf);initialized=true;
            }else feats_undistort->clear();
        }else imu.Process(meas,kf,feats_undistort);
        stages.imu_prepare_ms=imu.prepare_ms;stages.imu_predict_ms=imu.predict_ms;stages.deskew_ms=imu.deskew_ms;
        state_point=kf.get_x();pos_lid=state_point.pos+state_point.rot*state_point.offset_T_L_I;
        out.reason=1;
        if(feats_undistort->empty()){first_lidar_time=begin;imu.first_lidar_time=begin;}
        else {
            flg_EKF_inited=begin-first_lidar_time>=.1;
            lasermap_fov_segment();
            double downsample_begin=omp_get_wtime();
            downSizeFilterSurf.setInputCloud(feats_undistort);
            if(local_state_en){PointCloudXYZI::Ptr nearby(new PointCloudXYZI);
                for(const auto& p:*feats_undistort)if(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&
                    p.x*p.x+p.y*p.y+p.z*p.z<=local_max_range*local_max_range)nearby->push_back(p);
                downSizeFilterSurf.setInputCloud(nearby);}
            downSizeFilterSurf.filter(*feats_down_body);feats_down_size=feats_down_body->size();
            stages.downsample_ms=(omp_get_wtime()-downsample_begin)*1000;
            if(feats_down_size>100000)throw std::runtime_error("FAST-LIO workspace exhausted");
            feats_down_world->resize(feats_down_size);
            if(ikdtree.Root_Node==nullptr){
                if(feats_down_size>5){ikdtree.set_downsample_param(filter_size_map_min);
                    for(int i=0;i<feats_down_size;++i)pointBodyToWorld(&feats_down_body->points[i],&feats_down_world->points[i]);
                    ikdtree.Build(feats_down_world->points);
#ifdef SOCIAL_NAV_LIO_CUDA
                    if(cuda_matcher){cuda_world.clear();for(const auto& p:feats_down_world->points)cuda_world.push_back(gpu_point(p));cuda_matcher->reset(cuda_world.data(),cuda_world.size());}
#endif
                }
                out.reason=2;
            }else if(local_state_en&&feats_down_size<local_min_points)out.reason=3;
            else{
                normvec->resize(feats_down_size);Nearest_Points.resize(feats_down_size);
#ifdef SOCIAL_NAV_LIO_CUDA
                if(cuda_matcher){cuda_matcher->sync_map();cuda_body.clear();
                    for(const auto& p:feats_down_body->points)cuda_body.push_back(gpu_point(p));
                    cuda_matcher->begin_scan(cuda_body.data(),cuda_body.size());}
#endif
                double solve=0;kf.update_iterated_dyn_share_modified(.001,solve);state_point=kf.get_x();
                stages.filter_solve_ms=solve*1000;stages.jacobian_ms=solve_time*1000;
                pos_lid=state_point.pos+state_point.rot*state_point.offset_T_L_I;
                out.valid=flg_EKF_inited&&effct_feat_num>=local_min_points&&
                    double(effct_feat_num)/feats_down_size>=local_min_ratio&&std::isfinite(res_mean_last)&&res_mean_last<=local_max_error;
                out.reason=out.valid?0:4;
                if(!local_state_en||out.valid)map_incremental();
            }
        }
        Eigen::Matrix4d pose=Eigen::Matrix4d::Identity();
        pose.block<3,3>(0,0)=state_point.rot.toRotationMatrix()*state_point.offset_R_L_I.toRotationMatrix();
        pose.block<3,1>(0,3)=pos_lid;
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)out.world_from_lidar[i*4+j]=pose(i,j);
        for(int j=0;j<3;++j){out.velocity[j]=state_point.vel[j];out.gyro_bias[j]=state_point.bg[j];out.acc_bias[j]=state_point.ba[j];}
        out.residual=res_mean_last;out.effective_points=effct_feat_num;out.scan_points=feats_down_size;
        out.deskewed_points=feats_undistort->size();out.matching_ms=match_time*1000;
        Eigen::Matrix4d reference=Eigen::Matrix4d::Identity();
        if(adaptive_initialization&&initialized){
            reference.block<3,3>(0,0)=initializer.alignment*extrinsic_r;
            reference.block<3,1>(0,3)=initializer.alignment*extrinsic_t;
            out.reference_stamp=initializer.reference_stamp;
        }
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)out.initial_lidar_pose[4*i+j]=reference(i,j);
        out.map_update_ms=kdtree_incremental_time*1000;out.processing_ms=(omp_get_wtime()-start)*1000;
#ifdef SOCIAL_NAV_LIO_CUDA
        if(cuda_deskewer){auto s=cuda_deskewer->stats();stages.deskew_device_ms=s.deskew_device_ms;
            stages.deskew_h2d_bytes=s.h2d_bytes;stages.deskew_d2h_bytes=s.d2h_bytes;}
        if(cuda_matcher)stages.linearize_device_ms=cuda_matcher->stats().linearize_device_ms;
#endif
        return out.valid;
    }
    int cloud(float* out,size_t cap){size_t n=feats_undistort->size();if(cap<n)return -int(n);
        for(size_t i=0;i<n;++i){auto& p=feats_undistort->points[i];out[5*i]=p.x;out[5*i+1]=p.y;out[5*i+2]=p.z;out[5*i+3]=p.intensity;out[5*i+4]=p.curvature;}return n;}
    int acceleration(SnLioAcceleration& out){out={};
        int n=ikdtree.validnum();out.reference_points=n<0?0:n;
#ifdef SOCIAL_NAV_LIO_CUDA
        if(cuda_matcher){auto s=cuda_matcher->stats();out.map_points=s.points;out.h2d_bytes=s.h2d_bytes;out.d2h_bytes=s.d2h_bytes;
            out.nn_fallbacks=nn_fallbacks;out.plane_fallbacks=plane_fallbacks;out.map_sync_ms=s.map_ms;
            out.match_wall_ms=s.match_ms;out.query_device_ms=s.query_device_ms;out.backend=1;return 1;}
#endif
        out.map_points=out.reference_points;return 0;
    }
    int stage_stats(SnLioStages& out){out=stages;return 0;}
};

static thread_local std::string error;
extern "C" const char* sn_lio_error(){return error.c_str();}
extern "C" void* sn_lio_create(const char* path){try{return new Frontend(path);}catch(const std::exception& e){error=e.what();return nullptr;}}
extern "C" void sn_lio_destroy(void* p){delete static_cast<Frontend*>(p);}
extern "C" int sn_lio_step(void* p,const double* imu,size_t ni,const float* cloud,size_t n,double b,double e,SnLioResult* out){
    try{return static_cast<Frontend*>(p)->step(imu,ni,cloud,n,b,e,*out);}catch(const std::exception& ex){error=ex.what();return -1;}}
extern "C" int sn_lio_cloud(void* p,float* out,size_t n){return static_cast<Frontend*>(p)->cloud(out,n);}
extern "C" int sn_lio_acceleration(void* p,SnLioAcceleration* out){return static_cast<Frontend*>(p)->acceleration(*out);}
extern "C" int sn_lio_stages(void* p,SnLioStages* out){return static_cast<Frontend*>(p)->stage_stats(*out);}
