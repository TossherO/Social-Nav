# header.stamp is the matched scan END acquisition time, not publication time.
# header.frame_id is the continuous local world; pose maps child to that world.
std_msgs/Header header
string child_frame_id
geometry_msgs/Pose pose
bool has_covariance
# Row-major x,y,z,roll,pitch,yaw. Ignore the array when has_covariance is false.
float64[36] covariance
uint8 VALID=0
uint8 DEGRADED=1
uint8 INVALID=2
uint8 health
string status
string reason
string source
string session_id
uint64 generation
# Map pose may change at loop closure; it must not drive person velocities.
bool has_map_pose
string map_frame
geometry_msgs/Pose map_pose
uint64 map_revision
bool has_matching_quality
float64 inlier_ratio
float64 registration_error_m
