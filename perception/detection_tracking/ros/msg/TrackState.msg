uint8 OBSERVED=0
uint8 PREDICTED=1
string track_id
uint32 lifecycle_generation
uint8 state
bool confirmed
float64[8] box
bool has_velocity
float64[3] velocity_xyz_mps
# Row-major 10x10: x,y,z,dx,dy,dz,yaw,vx,vy,vz (mixed physical units).
float64[100] covariance
time last_observed_stamp
time last_supported_stamp
string support
string box_geometry
