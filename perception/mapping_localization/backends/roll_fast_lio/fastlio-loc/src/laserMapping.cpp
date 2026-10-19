// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <time.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include "MotionInitializer.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <social_nav_localization/LocalizationState.h>
#include <std_srvs/Trigger.h>
#include <limits>
//#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include<queue>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

nav_msgs::Path globalPathGPS;
#define pairPQ priority_queue< pair<double,int>,vector< pair<double,int> >,less< pair<double,int> >  >
Eigen::Affine3f affine_imu_to_lidar;
Eigen::Affine3f affine_lidar_to_gps;
Eigen::Affine3f affine_imu_to_body;
Eigen::Affine3f affine_imu_to_gps;
Eigen::Affine3f affine_lidar_to_body;
Eigen::Affine3f affine_gps_to_body;


std::vector<float>  array_imu_to_body;
std::vector<float>  array_lidar_to_body;
std::vector<float>  array_gps_to_body;


std::string gt_topic;
std::deque<std::vector<double>> gps_queue;
bool body_centered;
bool initialize_with_gt;

bool initialized = false;
bool info_filtering = false;

time_t now_time=time(NULL);  
tm*  t_tm = localtime(&now_time);   
long long time_sec = mktime(t_tm);
std::string local_time = to_string(time_sec);
std::string log_dir;

/*** Publishers ***/
ros::Subscriber sub_pcl;
ros::Subscriber sub_imu;
ros::Subscriber sub_gt;

ros::Publisher pubLaserCloudFull;
ros::Publisher pubLaserCloudFull_body;
ros::Publisher pubLaserCloudFull_lidar;
ros::Publisher pubLaserCloudEffect;
ros::Publisher pubLaserCloudMap;
ros::Publisher pubOdomAftMapped;
ros::Publisher pubOdomAftMappedLidar;
ros::Publisher pubPath;
ros::Publisher pubPathGPS;
ros::Publisher pubLocalState, pubLocalPath;
bool local_state_en = false, local_visualize = false, lidar_scan_pub_en = true;
int local_min_points = 20, local_path_limit = 2000;
int local_max_pending_scans = 32, local_max_pending_imu = 20000;
double local_min_ratio = 0.1, local_max_error = 0.2, local_max_range = 50.0;
std::string local_session;
nav_msgs::Path local_path;
double last_local_path_time = -1;

std::mutex getMap;
// std::mutex addIMU; // no need, because it is not multi-thread

// use imu as body frame
string imu_frame = "base_link";
bool debugMode;

// pcd loading
double  visualize_voxel_size;
bool map_pub;

/*** Log File Handler ***/
FILE *fp,*fp_time;
ofstream fout_dbg;

/*** variables definition ***/
int effect_feat_num = 0, frame_num = 0;
double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
bool flg_EKF_converged, EKF_stop_flg = 0;

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
std::ofstream frontend_metrics;
std::map<double, std::pair<double, double>> frontend_inputs;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
float lidar_delay = 0.0;
bool stamp_at_scan_end = true;
bool stationary_imu_assumption = false;
bool publish_tf = false;
double imu_time_offset = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_reset, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

// for loc
bool load_pcd;
vector<double> initial_pose(6,0.0);
string pcd_path;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));

pcl::VoxelGrid<PointType> downSizeFilterSurf;

KD_TREE ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);

V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

// shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<Preprocess> p_pre;
shared_ptr<ImuProcess> p_imu(new ImuProcess());
social_nav_init::MotionInitializer motion_initializer;
bool adaptive_initialization=true, motion_initialized=false;
ros::Publisher pubInitialLidarPose;

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;

int process_increments = 0;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{


    Eigen::Vector4d rot_quat = SO3toquaternion(state_point.rot);
    // V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    // fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    // fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));
    fprintf(fp, "%lf ", Measures.lidar_beg_time*1e+6);                   
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    // // tum format: xyz qw qx qy qz
    // fprintf(fp, "%lf %lf %lf %lf ", rot_quat(3), rot_quat(0), rot_quat(1),rot_quat(2));                   // Angle
    // nclt: xyz RPY
    V3D rot_euler = SO3ToEuler(state_point.rot);
    fprintf(fp, "%lf %lf %lf ", rot_euler(0)*M_PI/180,rot_euler(1)*M_PI/180,rot_euler(2)*M_PI/180);                   // Angle
    
    // fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    // fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    // fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_grav  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // Removed local-map points are discarded rather than archived without a bound.
}


void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (local_state_en && lidar_buffer.size() >= static_cast<size_t>(local_max_pending_scans)) {
        ROS_FATAL("Local odometry scan backlog exceeded its budget; restart after restoring input/timing");
        ros::shutdown();
        mtx_buffer.unlock();
        return;
    }
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
        time_buffer.clear();
        lidar_pushed = false;
        ROS_FATAL("LiDAR time reversed; restart the estimator for a new playback");
        ros::shutdown();
        mtx_buffer.unlock();
        return;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    const double preprocess_start = omp_get_wtime();
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    if (ptr->size() < 2) {
        lidar_buffer.pop_back();
        mtx_buffer.unlock();
        ROS_WARN_THROTTLE(1, "Skipping empty/invalid LiDAR scan");
        return;
    }
    const double scanDuration = ptr->points.back().curvature / 1000.0;
    const double scanStart = msg->header.stamp.toSec() - (stamp_at_scan_end ? scanDuration : 0.0);
    time_buffer.push_back(scanStart);
    if (frontend_metrics.is_open()) {
        frontend_inputs[scanStart] = {preprocess_start, (omp_get_wtime()-preprocess_start)*1000};
        while (frontend_inputs.size() > 128) frontend_inputs.erase(frontend_inputs.begin());
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
/*
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    // cout<<setiosflags(ios::fixed)<<setprecision(6)<<"lidar got at: "<<msg->header.stamp.toSec() + lidar_delay<<endl;
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec() + lidar_delay;
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
*/

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    // if(imu_frame == "") 
    // {
    //     imu_frame = msg_in->header.frame_id;
    //     cout<<"imu frame name: "<<imu_frame<<endl;
    // }

    publish_count ++;
    // cout<<setiosflags(ios::fixed)<<setprecision(6)<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    msg->header.stamp += ros::Duration(imu_time_offset);
    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (local_state_en && imu_buffer.size() >= static_cast<size_t>(local_max_pending_imu)) {
        ROS_FATAL("Local odometry IMU backlog exceeded its budget; restart after restoring LiDAR input");
        ros::shutdown();
        mtx_buffer.unlock();
        return;
    }
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
        if (local_state_en || adaptive_initialization) {
            ROS_FATAL("IMU time reversed; restart local odometry with a new session");
            ros::shutdown();
            mtx_buffer.unlock();
            return;
        }
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        if(meas.lidar->points.size() <= 1)
        {
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = time_buffer.front();
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    
    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}


void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();

// 2012-01-08-bin-start.bag error: solved by adding mutex lock
//     #12 0x00000000004389ef in map_incremental ()
//     at /home/haisenberg/Documents/ROLL/src/FAST_LIO/src/laserMapping.cpp:493
// 493	    add_point_size = ikdtree.Add_Points(PointToAdd, true);

    getMap.lock();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    getMap.unlock();

    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en && pubLaserCloudFull.getNumSubscribers())
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = local_state_en ? "odom" : "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    // why not save ikdtree output: featsFromMap
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    // if (pcd_save_en)
    // {
    //     int size = feats_undistort->points.size();
    //     PointCloudXYZI::Ptr laserCloudWorld( \
    //                     new PointCloudXYZI(size, 1));

    //     for (int i = 0; i < size; i++)
    //     {
    //         RGBpointBodyToWorld(&feats_undistort->points[i], \
    //                             &laserCloudWorld->points[i]);
    //     }
    //     *pcl_wait_save += *laserCloudWorld;

    //     static int scan_wait_num = 0;
    //     scan_wait_num ++;
    //     if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
    //     {
    //         pcd_index ++;
    //         string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
    //         pcl::PCDWriter pcd_writer;
    //         cout << "current scan saved to /PCD/" << all_points_dir << endl;
    //         pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    //         pcl_wait_save->clear();
    //         scan_wait_num = 0;
    //     }
    // }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    if (!pubLaserCloudFull_body.getNumSubscribers()) return;
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_frame_lidar(const ros::Publisher & pubLaserCloudFull_lidar)
{
    if (!pubLaserCloudFull_lidar.getNumSubscribers()) return;
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*feats_undistort, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "lidar";
    pubLaserCloudFull_lidar.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    if (!pubLaserCloudEffect.getNumSubscribers()) return;
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}



template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0); // imu 
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

bool publish_localization_state(const std::string& pending = "")
{
    if (!local_state_en) return true;
    social_nav_localization::LocalizationState message;
    message.header.stamp = ros::Time().fromSec(lidar_end_time);
    message.header.frame_id = "odom";
    message.child_frame_id = "rslidar";
    message.source = stationary_imu_assumption ? "fast_lio_stationary_assumption" : "fast_lio";
    message.session_id = local_session;
    message.has_covariance = false;
    message.has_map_pose = false;
    const SO3 rotation = state_point.rot * state_point.offset_R_L_I;
    message.pose.position.x = pos_lid.x();
    message.pose.position.y = pos_lid.y();
    message.pose.position.z = pos_lid.z();
    message.pose.orientation.x = rotation.coeffs()[0];
    message.pose.orientation.y = rotation.coeffs()[1];
    message.pose.orientation.z = rotation.coeffs()[2];
    message.pose.orientation.w = rotation.coeffs()[3];
    message.has_matching_quality = effct_feat_num > 0 && std::isfinite(res_mean_last);
    message.inlier_ratio = feats_down_size > 0 ? double(effct_feat_num) / feats_down_size : 0;
    message.registration_error_m = message.has_matching_quality ? res_mean_last : 0;
    std::string reason = pending;
    if (reason.empty() && !flg_EKF_inited) reason = "initializing";
    if (reason.empty() && (!pos_lid.allFinite() || !rotation.coeffs().allFinite())) reason = "nonfinite_pose";
    if (reason.empty() && (!message.has_matching_quality || effct_feat_num < local_min_points ||
        message.inlier_ratio < local_min_ratio || res_mean_last > local_max_error))
        reason = "local_registration_unavailable_or_failed";
    const bool valid = reason.empty();
    message.health = valid ? message.VALID : message.INVALID;
    const bool initializing = pending == "imu_initialization" || pending == "local_map_initialization" ||
                              reason == "initializing";
    message.status = valid ? "ODOMETRY" : (initializing ? "INITIALIZING" : "LOST");
    message.reason = reason;
    pubLocalState.publish(message);
    if (!valid) return false; // IMU prediction alone never becomes accepted localization or TF.
    static tf::TransformBroadcaster broadcaster;
    tf::Transform transform;
    tf::poseMsgToTF(message.pose, transform);
    broadcaster.sendTransform(tf::StampedTransform(transform, message.header.stamp, "odom", "rslidar"));
    if (local_visualize && pubLocalPath.getNumSubscribers() &&
        (last_local_path_time < 0 || lidar_end_time - last_local_path_time >= 0.2)) {
        geometry_msgs::PoseStamped pose;
        pose.header = message.header;
        pose.pose = message.pose;
        local_path.header = message.header;
        local_path.poses.push_back(pose);
        if (local_path.poses.size() > static_cast<size_t>(local_path_limit))
            local_path.poses.erase(local_path.poses.begin());
        pubLocalPath.publish(local_path);
        last_local_path_time = lidar_end_time;
    }
    return true;
}

void publish_odometry_lidar(const ros::Publisher & pubOdomAftMappedLidar)
{
    // cout<<"publishing odometry"<<endl;
    nav_msgs::Odometry odomAftMappedLidar;
    odomAftMappedLidar.header.frame_id = "camera_init";
    odomAftMappedLidar.child_frame_id = "lidar";
    odomAftMappedLidar.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    odomAftMappedLidar.pose.pose.position.x = pos_lid(0);  
    odomAftMappedLidar.pose.pose.position.y = pos_lid(1);
    odomAftMappedLidar.pose.pose.position.z = pos_lid(2);
    SO3 rot_lid = state_point.rot*state_point.offset_R_L_I;
    odomAftMappedLidar.pose.pose.orientation.x = rot_lid.coeffs()[0];
    odomAftMappedLidar.pose.pose.orientation.y = rot_lid.coeffs()[1];
    odomAftMappedLidar.pose.pose.orientation.z = rot_lid.coeffs()[2];
    odomAftMappedLidar.pose.pose.orientation.w = rot_lid.coeffs()[3];

    // convert vel and ang_vel to lidar frame
    M3D rot_R_I_L = state_point.offset_R_L_I.matrix().inverse(); //xyzw
    V3D vel_lidar = rot_R_I_L*state_point.vel;
    odomAftMappedLidar.twist.twist.linear.x = vel_lidar(0);
    odomAftMappedLidar.twist.twist.linear.y = vel_lidar(1);
    odomAftMappedLidar.twist.twist.linear.z = vel_lidar(2);

    if(Measures.imu.empty() == false)
    {
        sensor_msgs::Imu imu = *(Measures.imu.back());
        V3D imu_w = V3D(imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z);
        V3D lidar_w = rot_R_I_L*imu_w;
        odomAftMappedLidar.twist.twist.angular.x = lidar_w(0);
        odomAftMappedLidar.twist.twist.angular.y = lidar_w(1);
        odomAftMappedLidar.twist.twist.angular.z = lidar_w(2);
    }
    

    // cout<<"vel xyz: "<<vel_lidar(0)<<" "<<vel_lidar(1)<<" "<<vel_lidar(2)<<" yaw vel: "<<lidar_w(2)<<endl;
    pubOdomAftMappedLidar.publish(odomAftMappedLidar);

}
void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    // cout<<"publishing odometry"<<endl;
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    if (!publish_tf) return;
    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );

    tf::Transform unit_trans;
    unit_trans.setIdentity();
    // br.sendTransform( tf::StampedTransform( unit_trans, odomAftMapped.header.stamp, "body","imu_sensor_frame" ) );
    // br.sendTransform( tf::StampedTransform( unit_trans, odomAftMapped.header.stamp, "imu_sensor_frame",imu_frame ) );
    // br.sendTransform( tf::StampedTransform( unit_trans, odomAftMapped.header.stamp, "imu_adis","aux_imu_sensor_frame" ) );

    // publish lidar-imu tf
    state_ikfom state_cur = kf.get_x();
    V3D euler_cur = SO3ToEuler(state_cur.offset_R_L_I)*M_PI/180;
    tf::Transform tf_lidar_to_imu = tf::Transform(tf::createQuaternionFromRPY(euler_cur.x(),euler_cur.y(),euler_cur.z()),
                tf::Vector3(state_cur.offset_T_L_I.x(),state_cur.offset_T_L_I.y(),state_cur.offset_T_L_I.z()));
    // cout<<"extrensics euler "<<euler_cur.transpose()<<endl;
    br.sendTransform( tf::StampedTransform( tf_lidar_to_imu, odomAftMapped.header.stamp, "body","lidar" ) );


}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    // cout<<jjj<<endl;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        path.header.stamp = ros::Time().fromSec(lidar_end_time);
        path.header.frame_id = "camera_init";
        pubPath.publish(path);
    }
}

void pairPQoperation(pairPQ &pq, pair<double,int> &pairIn, int s_chosen){
    pq.push(pairIn);
    if(pq.size() > s_chosen) pq.pop();        return;
}


void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    // #ifdef MP_EN // executig once
    //     omp_set_num_threads(MP_PROC_NUM);
    //     #pragma omp parallel for
    // #endif

    // for saving 9 criterions
    const int value_num = 9;
    const int s_chosen = 30;
    vector<pairPQ> values; // only filter once maybe 
    for(int i=0;i<value_num;i++)
    {
        pairPQ pq;
        values.push_back(pq);
    }

    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            // different from pcl: points_near is point vector, not a index vector
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;

        V3D X(1.0,0.0,0.0);
        V3D Y(0.0,1.0,0.0);
        V3D Z(0.0,0.0,1.0);

        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float sF = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (sF > 0.9)
            {
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);

                // better after building a good map
                if(ikdtree.size() < 10000 || info_filtering == false) 
                {
                    point_selected_surf[i] = true;
                    // cout<<"A good map needs to be built before filtering even more"<<endl;
                }
                else
                {
                    // get 9 values for each point in lidar local frame, ignore the planeness
                    V3D normalGlobal(pabcd(0),pabcd(1),pabcd(2));
                    V3D normal(s.offset_R_L_I.conjugate() * s.rot.conjugate() * normalGlobal); // to lidar local 
                    vector<pair<double,int>> pairTmp;
                    // Vector3d is 3x1
                    pairTmp.push_back(make_pair((normal.cross(p_body)).transpose()*X,i));
                    pairTmp.push_back(make_pair((-normal.cross(p_body)).transpose()*X,i));
                    pairTmp.push_back(make_pair((normal.cross(p_body)).transpose()*Y,i));
                    pairTmp.push_back(make_pair((-normal.cross(p_body)).transpose()*Y,i));
                    pairTmp.push_back(make_pair((normal.cross(p_body)).transpose()*Z,i));
                    pairTmp.push_back(make_pair((-normal.cross(p_body)).transpose()*Z,i));

                    pairTmp.push_back(make_pair(fabs(normal.transpose()*X),i));
                    pairTmp.push_back(make_pair(fabs(normal.transpose()*Y),i));
                    pairTmp.push_back(make_pair(fabs(normal.transpose()*Z),i));
                    
                    for(int j=0; j<pairTmp.size(); j++) pairPQoperation(values[j],pairTmp[j],s_chosen);
                }                

            }
        }
    }

    // set point_selected_surf to true for the value remaining in pairPQ
    if(ikdtree.size() >= 10000 && info_filtering == true) 
    {
        for(int i=0; i<value_num; i++)
        {
            while(values[i].empty() == false)
            {
                point_selected_surf[values[i].top().second] = true;
                values[i].pop();
            }
        }
    }


    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num == 0) {
        res_mean_last = std::numeric_limits<double>::infinity();
        ekfom_data.valid = false;
        return;
    }
    res_mean_last = total_residual / effct_feat_num;
    if (local_state_en && (effct_feat_num < local_min_points ||
        double(effct_feat_num)/feats_down_size < local_min_ratio ||
        !std::isfinite(res_mean_last) || res_mean_last > local_max_error)) {
        ekfom_data.valid = false;
        return;
    }
    match_time  += omp_get_wtime() - match_start;


    double solve_start_  = omp_get_wtime();   
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    if(debugMode) cout<<"Out of all feature points "<<feats_down_size<<" inlier num: "<<effct_feat_num<<"residual mean: "<<res_mean_last<<endl;

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_; 
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
void gt_handler(const nav_msgs::Odometry::ConstPtr& gtMsg)
{
    if ( isnan(gtMsg->pose.pose.position.x) || isnan(gtMsg->pose.pose.position.y) || isnan(gtMsg->pose.pose.position.z)
    || gtMsg->pose.covariance[0] > 1.0)
    {
        // if(debugMode) cout<<"x: "<< gtMsg->pose.pose.position.x<<" cov: "<<gtMsg->pose.covariance[0]<<endl;
        return;
    }
    Eigen::Affine3f affine_sth_to_map,affine_lidar_to_map;
    odometryMsgToAffine3f(*gtMsg, affine_sth_to_map);
    //for lab datasets, gt is trans_gps_to_enu, we use enu as map
    // Eigen::Affine3f affine_lidar_to_body = pcl::getTransformation(0.002, -0.004, -0.957,
    //     0.014084807063594,0.002897246558311,-1.583065991436417);
    // Eigen::Affine3f affine_lidar_to_gps = pcl::getTransformation(0,0,0,0,0,0.06);
    if(body_centered == false)
    {
        affine_lidar_to_map = affine_sth_to_map*affine_imu_to_gps;
    }
    else
    {
        // for nclt, gt is trans_body_to_ned, we use NED as map
        affine_lidar_to_map = affine_sth_to_map*affine_imu_to_body;
    }

    float roll,pitch,yaw,x,y,z;
    // no need for z roll pitch
    // if they are really needed, try set lidar height as the initial height
    pcl::getTranslationAndEulerAngles(affine_lidar_to_map,x,y,z,roll,pitch,yaw);
    tf::Quaternion q = tf::createQuaternionFromRPY(roll,pitch,yaw);
    geometry_msgs::PoseStamped pose_stamped_gps;
    pose_stamped_gps.header.frame_id = "camera_init";
    pose_stamped_gps.header.stamp = gtMsg->header.stamp;
    pose_stamped_gps.pose.position.x = x;
    pose_stamped_gps.pose.position.y = y;
    pose_stamped_gps.pose.position.z = z;
    pose_stamped_gps.pose.orientation.x = q.x();
    pose_stamped_gps.pose.orientation.y = q.y();
    pose_stamped_gps.pose.orientation.z = q.z();
    pose_stamped_gps.pose.orientation.w = q.w(); 

    static long cnt_gps = 0;
    cnt_gps++;
    
    if(cnt_gps%10 == 0 &&  pubPathGPS.getNumSubscribers() != 0)
    {
        globalPathGPS.poses.push_back(pose_stamped_gps);
        globalPathGPS.header = pose_stamped_gps.header;
        pubPathGPS.publish(globalPathGPS);
    }
    std::vector<double> gt_pose={gtMsg->header.stamp.toSec(),roll,pitch,yaw,x,y,z};
    gps_queue.push_back(gt_pose);
}

struct FrontendTiming {
    double start = omp_get_wtime();
    double preprocessMs = std::numeric_limits<double>::quiet_NaN();
    double waitMs = std::numeric_limits<double>::quiet_NaN();
    explicit FrontendTiming(double stamp) {
        if (!frontend_metrics.is_open()) return;
        auto input = frontend_inputs.find(stamp);
        if (input != frontend_inputs.end()) {
            preprocessMs = input->second.second;
            waitMs = (start-input->second.first)*1000-preprocessMs;
            frontend_inputs.erase(frontend_inputs.begin(), std::next(input));
        }
    }
    ~FrontendTiming() {
        if (!frontend_metrics.is_open()) return;
        frontend_metrics << std::setprecision(18) << lidar_end_time << ','
            << ros::Time().fromSec(lidar_end_time).toNSec() << ','
            << (omp_get_wtime()-start)*1000 << ',' << match_time*1000 << ','
            << kdtree_incremental_time*1000 << ',' << (ros::Time::now().toSec()-lidar_end_time)*1000 << ','
            << preprocessMs << ',' << waitMs << '\n';
    }
};

void run()
{
    ros::NodeHandle nh;

//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    // Poll sensors and honor SIGINT even while a replay clock is paused.
    ros::WallRate rate(5000);
    while (ros::ok())
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            FrontendTiming timing(Measures.lidar_beg_time);
            match_time = kdtree_incremental_time = 0;
            effct_feat_num = 0;
            feats_down_size = 0;
            res_mean_last = std::numeric_limits<double>::infinity();
            if (flg_reset)
            {
                if (adaptive_initialization) {
                    ROS_ERROR("Time reversed during adaptive initialization/odometry; restart the session");
                    ros::shutdown(); break;
                }
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                Measures.imu.clear();
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            if(initialize_with_gt && initialized == false)
            {
                // although fastlio projects all points in a scan to the scan end time
                while(!gps_queue.empty() && abs((gps_queue.front())[0] - Measures.lidar_beg_time) > 0.1 )
                    gps_queue.pop_front();
                if(gps_queue.empty()) continue;
                
                initialized = true;
                for(int i=0;i<6;i++) {
                    initial_pose[i] = gps_queue.front()[i+1];
                }
                std::cout<<"initialized with gt"<<std::endl;
                for(auto i:initial_pose) std::cout<<" "<<i;
                std::cout<<std::endl;
            }

            if (adaptive_initialization && !motion_initialized) {
                if (!motion_initializer.process(Measures.lidar,Measures.lidar_beg_time,Measures.imu,*feats_undistort)) {
                    feats_undistort->clear();
                    ROS_INFO_THROTTLE(1.0,"Initialization: %s, %.2fs, %d scans, RMS %.3f, |g_raw| %.3f, |b_a| %.3f, tilt sensitivity %.4frad, stable %.2fs",
                        motion_initializer.reason.c_str(),motion_initializer.duration,motion_initializer.scan_count,
                        motion_initializer.result.rms,motion_initializer.result.gravity_raw_norm,
                        motion_initializer.result.acc_bias.norm(),motion_initializer.result.tilt_sensitivity,
                        motion_initializer.readiness.stable_duration);
                } else {
                    auto state=kf.get_x();
                    state.rot=motion_initializer.rotation_end;
                    state.pos=motion_initializer.position_end;
                    state.vel=motion_initializer.result.velocity;
                    state.bg=motion_initializer.result.gyro_bias;
                    state.ba=motion_initializer.result.acc_bias;
                    state.grav=S2(motion_initializer.result.gravity);
                    state.offset_R_L_I=Lidar_R_wrt_IMU; state.offset_T_L_I=Lidar_T_wrt_IMU;
                    p_imu->initializeMotion(Measures,state,kf);
                    motion_initialized=true;
                    geometry_msgs::PoseStamped reference;
                    reference.header.stamp=ros::Time().fromSec(motion_initializer.reference_stamp);
                    reference.header.frame_id="camera_init";
                    const V3D p=motion_initializer.alignment*Lidar_T_wrt_IMU;
                    const Eigen::Quaterniond q(motion_initializer.alignment*Lidar_R_wrt_IMU);
                    reference.pose.position.x=p.x(); reference.pose.position.y=p.y(); reference.pose.position.z=p.z();
                    reference.pose.orientation.x=q.x(); reference.pose.orientation.y=q.y();
                    reference.pose.orientation.z=q.z(); reference.pose.orientation.w=q.w();
                    pubInitialLidarPose.publish(reference);
                    nh.setParam("initialization/result/mode",motion_initializer.mode);
                    nh.setParam("initialization/result/duration_s",motion_initializer.duration);
                    nh.setParam("initialization/result/position_rms_m",motion_initializer.result.rms);
                    nh.setParam("initialization/result/reference_stamp_s",motion_initializer.reference_stamp);
                    nh.setParam("initialization/result/ready_stamp_s",motion_initializer.end_stamp);
                    nh.setParam("initialization/result/gyro_bias",std::vector<double>(state.bg.data(),state.bg.data()+3));
                    nh.setParam("initialization/result/acc_bias",std::vector<double>(state.ba.data(),state.ba.data()+3));
                    nh.setParam("initialization/result/velocity",std::vector<double>(state.vel.data(),state.vel.data()+3));
                    nh.setParam("initialization/result/accepted_scans",motion_initializer.scan_count);
                    nh.setParam("initialization/result/imu_samples",motion_initializer.sample_count);
                    nh.setParam("initialization/result/stable_duration_s",motion_initializer.readiness.stable_duration);
                    if (motion_initializer.mode=="moving")
                        nh.setParam("initialization/result/tilt_sensitivity_rad",motion_initializer.result.tilt_sensitivity);
                    nh.setParam("initialization/result/acc_bias_model","axial_only; perpendicular components remain uncertain");
                    ROS_INFO("%s initialization ready: %.2fs, RMS %.4fm, speed %.3fm/s, bias_a %.3f %.3f %.3f",
                        motion_initializer.mode.c_str(),motion_initializer.duration,motion_initializer.result.rms,
                        state.vel.norm(),state.ba[0],state.ba[1],state.ba[2]);
                }
            } else if(load_pcd)
                p_imu->Process(initial_pose, Measures, kf, feats_undistort);
            else
                p_imu->Process(Measures, kf, feats_undistort);

            state_point = kf.get_x();
            if(debugMode) cout<<"current x y z "<<state_point.pos<<endl;
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            // where is feats_undistort assigned value?
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                publish_localization_state("imu_initialization");
                // cout<<"FAST-LIO not ready"<<endl;
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            if(debugMode) cout<<"point cloud size: "<<feats_undistort->size()<<endl;
            downSizeFilterSurf.setInputCloud(feats_undistort);
            if (local_state_en) {
                // Bound the observed region as well as the moving ikd-Tree cube.
                PointCloudXYZI::Ptr nearby(new PointCloudXYZI());
                nearby->reserve(feats_undistort->size());
                for (const auto& point : feats_undistort->points)
                    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
                        point.x*point.x + point.y*point.y + point.z*point.z <= local_max_range*local_max_range)
                        nearby->push_back(point);
                downSizeFilterSurf.setInputCloud(nearby);
            }
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();

            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                publish_localization_state("local_map_initialization");
                continue;
            }
            if (local_state_en && feats_down_size < local_min_points) {
                publish_localization_state("insufficient_scan_points");
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            if(debugMode) cout<<"map num: "<<featsFromMapNum<<endl;
            /*** ICP and iterated Kalman filter update ***/
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;

            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            if(debugMode) cout<<"after opt. x y z "<<state_point.pos<<endl;

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            publish_odometry_lidar(pubOdomAftMappedLidar);
            const bool accepted = publish_localization_state();

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            if (accepted) map_incremental();
            t5 = omp_get_wtime();
            
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (accepted && (scan_pub_en || pcd_save_en)) publish_frame_world(pubLaserCloudFull);
            if (scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);
            if (lidar_scan_pub_en)
                publish_frame_lidar(pubLaserCloudFull_lidar);
            publish_effect_world(pubLaserCloudEffect);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;

                time_log_counter ++;
                // solve_time only takes 1ms...
                // fprintf(fp_time,"%0.8f %0.8f %f %0.8f %d %d\n",Measures.lidar_beg_time,aver_time_icp,aver_time_match,aver_time_solve,(int)effct_feat_num,(int)feats_undistort->points.size());

                // printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                // V3D rot = SO3ToEuler(state_point.rot);
                // fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << rot.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                if(log_dir != "") dump_lio_state_to_log(fp);
            }
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                if(debugMode) 
                    fout_dbg<< setw(20) <<Measures.lidar_beg_time - first_lidar_time<<" " <<ext_euler.transpose() 
                        << " "<<state_point.offset_T_L_I.transpose()<<" "<<feats_undistort->size()<<" "
                        <<feats_down_size<<" "<<featsFromMapNum<<" "<<add_point_size<<"\n";
        }

        rate.sleep();
    }
}

void publishMapThread()
{
    ros::WallRate rate(0.2);
    while(ros::ok())
    {
        if(ikdtree.Root_Node != nullptr && map_pub && pubLaserCloudMap.getNumSubscribers() != 0)
        {
            double t_m1 = omp_get_wtime();
            getMap.lock();
            PointVector ().swap(ikdtree.PCL_Storage);
            ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
            featsFromMap->clear();
            featsFromMap->points = ikdtree.PCL_Storage;
            // pcl::toROSMsg(*featsFromMap, laserCloudMap); // it is okay to not lock?
            getMap.unlock();

            pcl::VoxelGrid<PointType> mapFilter; // for global map visualization
            PointCloudXYZI::Ptr featsFromMapDS(new PointCloudXYZI());
            mapFilter.setLeafSize(visualize_voxel_size, visualize_voxel_size, visualize_voxel_size); // for global map visualization
            mapFilter.setInputCloud(featsFromMap);
            mapFilter.filter(*featsFromMapDS);   
            // cout<<"after filtering: "<<featsFromMapDS->size()<<endl;
            sensor_msgs::PointCloud2 laserCloudMap;
            pcl::toROSMsg(*featsFromMapDS, laserCloudMap); // it is okay to not lock?
            laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
            laserCloudMap.header.frame_id = "camera_init";
            pubLaserCloudMap.publish(laserCloudMap);
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;
            double t_m2 = omp_get_wtime();
            if(t_m2-t_m1 > 100) cout<<"warning: publishing map takes more than 100 ms"<<endl;
            // cout<<"publishing map takes "<<t_m2-t_m1<<" sec"<<endl;
        }
        rate.sleep();
    }
}

Eigen::Affine3f trans2Affine3f(const vector<float> &transformIn)
{
    if(transformIn.size() != 6) return Eigen::Affine3f::Identity();
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    p_pre = make_shared<Preprocess>(nh);
    nh.param<bool>("stationary_imu/enabled", stationary_imu_assumption, false);

    nh.param<vector<float>>("array_imu_to_body", array_imu_to_body, vector<float>(6, 0));
    nh.param<vector<float>>("array_lidar_to_body", array_lidar_to_body, vector<float>(6, 0));
    nh.param<vector<float>>("array_gps_to_body", array_gps_to_body, vector<float>(6, 0));

    affine_imu_to_body = trans2Affine3f(array_imu_to_body);
    affine_lidar_to_body =  trans2Affine3f(array_lidar_to_body);
    affine_gps_to_body =  trans2Affine3f(array_gps_to_body);

    affine_imu_to_lidar = affine_lidar_to_body.inverse()*affine_imu_to_body;
    affine_imu_to_gps = affine_gps_to_body.inverse()*affine_imu_to_body;
    
    affine_lidar_to_gps = affine_gps_to_body.inverse()*affine_lidar_to_body;


    nh.param<bool>("debugMode",debugMode, false);
    if(debugMode)
    {
        cout<<"affine_lidar_to_gps: "<<affine_lidar_to_gps.matrix()<<endl;
        cout<<"affine_imu_to_gps: "<<affine_imu_to_gps.matrix()<<endl;
    }
    
    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<bool>("publish/map_pub",map_pub, false);
    nh.param<bool>("publish/lidar_scan_en", lidar_scan_pub_en, scan_pub_en && scan_body_pub_en);
    nh.param<bool>("odometry/publish_state", local_state_en, false);
    nh.param<bool>("odometry/visualize", local_visualize, false);
    nh.param<int>("odometry/min_effective_points", local_min_points, 20);
    nh.param<double>("odometry/min_inlier_ratio", local_min_ratio, 0.1);
    nh.param<double>("odometry/max_registration_error_m", local_max_error, 0.2);
    nh.param<double>("odometry/max_range", local_max_range, 50.0);
    nh.param<int>("odometry/visualization_path_limit", local_path_limit, 2000);
    nh.param<int>("odometry/max_pending_scans", local_max_pending_scans, 32);
    nh.param<int>("odometry/max_pending_imu", local_max_pending_imu, 20000);
    if (local_min_points < 1 || local_path_limit < 1 || local_max_pending_scans < 1 ||
        local_max_pending_imu < 1 || !std::isfinite(local_min_ratio) ||
        local_min_ratio <= 0 || local_min_ratio > 1 || !std::isfinite(local_max_error) ||
        local_max_error <= 0 || !std::isfinite(local_max_range) || local_max_range <= 0) {
        ROS_FATAL("Invalid local odometry budget or registration thresholds");
        return 1;
    }

    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);

    nh.param<float>("common/lidar_delay", lidar_delay, 0);

    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);

    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("preprocess/stamp_at_scan_end", stamp_at_scan_end, true);
    nh.param<bool>("publish/publish_tf", publish_tf, false);
    nh.param<double>("common/imu_time_offset", imu_time_offset, 0.0);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    bool write_details;
    nh.param<bool>("odometry/writeDetails", write_details, false);
    if (write_details) {
        std::string directory;
        nh.param<std::string>("log_dir", directory, "");
        frontend_metrics.open(directory + "/frontend_timings.csv");
        if (!frontend_metrics) throw std::runtime_error("Cannot open FAST-LIO timing output");
        frontend_metrics << "stamp,stamp_ns,fast_lio_processing_ms,fast_lio_matching_ms,fast_lio_map_update_ms,"
                            "fast_lio_age_ms,fast_lio_preprocess_ms,fast_lio_input_wait_ms\n";
    }

    nh.param<bool>("need_map_erected", p_imu->need_map_erected, true);
    

    nh.param<bool>("body_centered", body_centered, true);
    nh.param<bool>("initialize_with_gt", initialize_with_gt, false);
    p_imu->gt_available = initialize_with_gt;
    nh.param<bool>("initialization/enabled",adaptive_initialization,false);
    auto& init=motion_initializer.options;
    nh.param<double>("initialization/min_duration",init.min_duration,1.0);
    nh.param<double>("initialization/max_duration",init.max_duration,6.0);
    nh.param<int>("initialization/min_scans",init.min_scans,10);
    nh.param<double>("initialization/acceleration_scale",init.acceleration_scale,1.0);
    nh.param<double>("initialization/voxel",init.voxel,.3);
    nh.param<double>("initialization/max_residual",init.max_residual,.06);
    nh.param<double>("initialization/max_acc_bias",init.max_acc_bias,1.5);
    nh.param<double>("initialization/max_tilt_sensitivity",init.max_tilt_sensitivity,.01);
    nh.param<double>("initialization/stable_duration",init.stable_duration,.3);
    if (!std::isfinite(init.acceleration_scale) || init.acceleration_scale<=0 ||
        !std::isfinite(init.min_duration) || init.min_duration<.5 ||
        !std::isfinite(init.max_duration) || init.max_duration<init.min_duration || init.max_duration>10 ||
        init.min_scans<8 || init.min_scans>100 || !std::isfinite(init.voxel) || init.voxel<=0 ||
        !std::isfinite(init.max_residual) || init.max_residual<=0 ||
        !std::isfinite(init.max_acc_bias) || init.max_acc_bias<=0 ||
        !std::isfinite(init.max_tilt_sensitivity) || init.max_tilt_sensitivity<=0 ||
        !std::isfinite(init.stable_duration) || init.stable_duration<.1 || init.stable_duration>init.max_duration) {
        ROS_FATAL("Invalid adaptive initialization parameters"); return 1;
    }
    p_imu->fixed_acceleration_scale=adaptive_initialization;
    p_imu->acceleration_scale=init.acceleration_scale;
    if (adaptive_initialization) nh.deleteParam("initialization/result");

    nh.param<bool>("mapping/runtime_pos_log_enable", runtime_pos_log, false);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);

    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    // loading initial pose from config file
    nh.param<bool>("mapping/load_pcd", load_pcd, false);
    nh.param<string>("gt_topic", gt_topic, "ground_truth");
    nh.param<string>("log_dir", log_dir, "");
    nh.param<double>("mapping/visualize_voxel_size", visualize_voxel_size, 1.0);

    if(load_pcd)
    {
        nh.param<string>("pcd_path", pcd_path,"");
        if(!initialize_with_gt)    nh.param<vector<double>>("mapping/initial_pose", initial_pose, vector<double>(6,0));
        
    }
        
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    cout<<"load pcd: "<<load_pcd<<endl;
    cout<<"map pub: "<<map_pub<<endl;
    cout<<"path_en"<<path_en<<endl;
    cout<<"pose output activated: "<<runtime_pos_log<<endl;

    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);


    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    if (extrinT.size() != 3 || extrinR.size() != 9) {
        ROS_FATAL("Invalid LiDAR-IMU extrinsic dimensions"); return 1;
    }
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    motion_initializer.setExtrinsic(Lidar_R_wrt_IMU,Lidar_T_wrt_IMU);
    if (adaptive_initialization && (load_pcd || initialize_with_gt || !p_imu->need_map_erected)) {
        ROS_FATAL("Adaptive initialization requires gravity-aligned local operation without load_pcd/GT; use initialization/enabled=false for legacy operation");
        return 1;
    }
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    std::string add_prefix = "";
    if(initialize_with_gt)
        add_prefix = local_time;

    string pos_log_dir = log_dir + "/logs/pos_log"+add_prefix+".txt";
    if(log_dir != "")
    {
        if (runtime_pos_log) {
            fp = fopen(pos_log_dir.c_str(),"w");
            if (fp) cout <<pos_log_dir<<" file opened" << endl;
            else cout <<pos_log_dir<<" cannot be created" << endl;
        }
        if(debugMode)
        {
          fout_dbg.open(log_dir + "/debug.txt",ios::out);
        }
         

    } 

    /*** ROS subscribe initialization ***/
    sub_gt   = nh.subscribe<nav_msgs::Odometry> (gt_topic, 200, gt_handler); 
    sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 64, standard_pcl_cbk) : \
        nh.subscribe(lid_topic, 64, standard_pcl_cbk);
    sub_imu = nh.subscribe(imu_topic, 20000, imu_cbk);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 8);
    pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 8);
    pubLaserCloudFull_lidar = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_lidar", 8);
    pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 8);
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 8);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 8);
    pubOdomAftMappedLidar = nh.advertise<nav_msgs::Odometry> 
            ("/OdometryLidar", 8);
    pubInitialLidarPose=nh.advertise<geometry_msgs::PoseStamped>("/fast_lio/initial_lidar_pose",1,true);
    pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 8);
    pubPathGPS    = nh.advertise<nav_msgs::Path>("/gnss", 10000);
    ros::ServiceServer local_status;
    if (local_state_en) {
        local_session = "fast_lio_" + std::to_string(ros::WallTime::now().toNSec());
        pubLocalState = nh.advertise<social_nav_localization::LocalizationState>("/localization/state", 8);
        pubLocalPath = nh.advertise<nav_msgs::Path>("/localization/path", 1);
        local_status = nh.advertiseService<std_srvs::Trigger::Request, std_srvs::Trigger::Response>(
            "/fast_lio/status", +[](std_srvs::Trigger::Request&, std_srvs::Trigger::Response& response) {
                response.success = true;
                response.message = "ready_for_lidar_and_imu";
                return true;
            });
    }

    // loading pcd into ikdtree
    if(load_pcd)
    {
//        ' normal_z'.
// Failed to find match for field 'curvature'
        cout<<"Loading pcd from path: "<<pcd_path<<endl;
        ikdtree.set_downsample_param(filter_size_map_min);
        feats_down_world->resize(feats_down_size);
        PointCloudXYZI::Ptr preloadedMap(new PointCloudXYZI());  
        pcl::io::loadPCDFile<PointType> (pcd_path, *preloadedMap);
        ikdtree.Build(preloadedMap->points);
        cout<<"ikdtree size "<<ikdtree.size()<<endl; 

                // only visualize once
        double t_m1 = omp_get_wtime();

        getMap.lock();
        PointVector ().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;
        sensor_msgs::PointCloud2 laserCloudMap;

        pcl::VoxelGrid<PointType> mapFilter; // for global map visualization
        PointCloudXYZI::Ptr featsFromMapDS(new PointCloudXYZI());
        mapFilter.setLeafSize(visualize_voxel_size, visualize_voxel_size, visualize_voxel_size); // for global map visualization
        mapFilter.setInputCloud(featsFromMap);
        mapFilter.filter(*featsFromMapDS);   
        cout<<"after filtering: "<<featsFromMapDS->size()<<endl;
        pcl::toROSMsg(*featsFromMapDS, laserCloudMap); // it is okay to not lock?
        getMap.unlock();

        laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudMap.header.frame_id = "camera_init";
        pubLaserCloudMap.publish(laserCloudMap);
        // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;
        double t_m2 = omp_get_wtime();
        cout<<"publishing map takes "<<t_m2-t_m1<<" sec"<<endl;

    }

    // threads
    std::thread odometry_thread(run);
    std::thread vis_map_thread;
    if (map_pub) vis_map_thread = std::thread(publishMapThread);
    odometry_thread.join();
    ros::shutdown();
    if (vis_map_thread.joinable()) vis_map_thread.join();
    // run();

    // fout_out.close();
    // fout_pre.close();
    // it won't be displayed if you ctrl+c, but "rosnode kill -a" can

    /**************** save map ****************/
    if (pcd_save_en && !featsFromMap->empty())
    {
        string file_name = string("map.pcd");
        // string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        // cout << "current scan saved to /PCD/" << file_name<<endl;
        pcl::VoxelGrid<PointType> mapFilter; // for global map visualization
        PointCloudXYZI::Ptr featsFromMapDS(new PointCloudXYZI());
        mapFilter.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min); // for global map visualization
        mapFilter.setInputCloud(featsFromMap);
        mapFilter.filter(*featsFromMapDS);   
        cout<<"after filtering: "<<featsFromMapDS->size()<<endl;
        pcd_writer.writeBinary(pcd_path, *featsFromMapDS);
    }
    cout<<"laserMapping process is over "<<endl; 

    return 0;
}
