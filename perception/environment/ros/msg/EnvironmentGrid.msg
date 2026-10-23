# All arrays are row-major, with info.width * info.height entries.
# Missing scalar geometry is NaN; occupancy=-1 remains unknown.
std_msgs/Header header
string snapshot_id
string session_id
uint64 generation
nav_msgs/MapMetaData info
int8[] occupancy
float32[] ground_z
uint8[] ground_support
uint8[] observed
float64[] last_observed
float32[] coverage
float32[] obstacle_min_z
float32[] obstacle_max_z
float32[] clearance
float32[] unknown_distance
uint8[] frontier
bool rays_used
string geometry_status
string association_json
string metadata_json
