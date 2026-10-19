## Environments:
Ubuntu 16.04, ROS kinetic
PCL 1.7
Eigen 3.3.7
GTSAM 4.0.2 (-DGTSAM_BUILD_WITH_MARCH_NATIVE = OFF)

## Batch processing
1. for log, prepara keyframe dir & bag files in bag_dir
2. batch_eval bag_dir, then keyframe dir will be updated every time there is a tm in a sequence evaluation

## branches
surf_loc: current main branch, developing imu+lidar+gps mapping




