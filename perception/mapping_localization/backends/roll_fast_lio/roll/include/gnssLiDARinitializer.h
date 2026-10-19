/*
 * @brief r=R(theta)*p-p'
 * p: receiver position i with respect to receiver frame 0
 * p': frame position with respect to enu frame
 */

#include"utility.h"

// Assume the first LLA is accurate for anchor fixation
class gnssLiDARinitializer
{
    public:
        const int num_frames;
        int frame_num_cur = 0;
        bool initialized = false;

        Eigen::MatrixXd mat_roll;
        Eigen::MatrixXd mat_gnss;

        Eigen::Matrix4d lio_init;
        Eigen::Matrix4d gnss_init;

        Eigen::Affine3d affine_odom_to_enu;
        double last_lio_time = -1;

        gnssLiDARinitializer(const int n):num_frames(n)
        {
            // obs_vec.resize(num_frames*2);
            mat_roll = Eigen::MatrixXd::Zero(2,num_frames);
            mat_gnss = Eigen::MatrixXd::Zero(2,num_frames);
        }

        void getSyncFrames(double lio_time, Eigen::Affine3f affine_lidar_to_lidar0,Eigen::Affine3f affine_lidar_to_enu)
        {
            if(last_lio_time != -1 && lio_time-last_lio_time > 0.1)
            {
                // std::cout<<"time window is too short !"<<std::endl;
                last_lio_time = lio_time;
                return;
            }
            last_lio_time = lio_time;
            frame_num_cur++;

            if(frame_num_cur == 1)
            {
                gnss_init = affine_lidar_to_enu.matrix().cast<double>();
                lio_init = affine_lidar_to_lidar0.matrix().cast<double>();  
            }

            Eigen::Vector2d p_roll;
            Eigen::Matrix4d affine_lidar_to_init = lio_init.inverse()*(affine_lidar_to_lidar0.matrix().cast<double>());
            p_roll<<(double)affine_lidar_to_init(0,3),(double)affine_lidar_to_init(1,3);
            Eigen::Vector2d gnss_cur;
            gnss_cur<<(double)affine_lidar_to_enu(0,3),(double)affine_lidar_to_enu(1,3);
            Eigen::Vector2d p_gnss=gnss_cur-gnss_init.block<2,1>(0,3);
            mat_roll.col(frame_num_cur-1) = p_roll;
            mat_gnss.col(frame_num_cur-1) = p_gnss;
            std::cout<<"frame pair "<<frame_num_cur<<std::endl;
        }

        bool tryInitializing()
        {
            if (frame_num_cur == num_frames)
            {
                std::cout<<"start umeyama method"<<std::endl;
                
                // Umeyama method
                Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
                Eigen::Matrix2d Ryaw = Eigen::Matrix2d::Zero();

                for(int i=0;i<num_frames;i++)
                {                
                    Eigen::Matrix2d tmpH = H;
                    H = tmpH + mat_roll.col(i)*(mat_gnss.col(i).transpose());
                }
                Eigen::Matrix2d tmpH = H;
                H = tmpH/num_frames;
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::Matrix2d V = svd.matrixV(), U = svd.matrixU();
                // Eigen::Matrix2d  D = U.inverse() * H * V.transpose().inverse(); 
                Ryaw = V*(U.transpose());

                Eigen::Matrix4d affine_lidar0_to_enu = Eigen::Matrix4d::Identity();
                affine_lidar0_to_enu.block<3,1>(0,3) = gnss_init.block<3,1>(0,3);
                affine_lidar0_to_enu.block<2,2>(0,0) = Ryaw;
                affine_odom_to_enu = affine_lidar0_to_enu*lio_init.inverse().matrix();
            
                std::cout<<"Ryaw: "<<Ryaw<<std::endl;
                Eigen::MatrixXd errorMat =Eigen::MatrixXd::Zero(2,num_frames);
                errorMat = Ryaw*mat_roll-mat_gnss;
                double error_norm = errorMat.norm();
                std::cout<<"error mat norm: "<<error_norm<<std::endl;
                initialized = true;
                return true;                   
            }
            return false;
        }
        Eigen::Affine3f getTer()
        {
            if(initialized)
            {
                return affine_odom_to_enu.cast<float>();
            }
            else return Eigen::Affine3f::Identity();
        }
};

