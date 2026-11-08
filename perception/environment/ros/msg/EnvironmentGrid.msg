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
# Optional local ground conflicts. A marked cell cannot use historical free support.
uint8[] ground_conflict
uint8[] observed
float64[] last_observed
float32[] coverage
float32[] obstacle_min_z
float32[] obstacle_max_z
float32[] clearance
float32[] unknown_distance
uint8[] frontier
# Optional row-major history provenance; empty when the local policy is disabled.
uint8[] near_field_free
float64[] near_field_free_classified_at
uint8[] near_field_obstacle
float64[] near_field_obstacle_stamp
float64[] near_field_obstacle_expires_at
# Optional provenance: occupied exclusively by reliable person evidence on a
# previously certified free background. Current occupancy remains occupied.
uint8[] person_background_free
# Optional derived navigation grid. Raw occupancy and measurement times stay intact.
int8[] navigation_occupancy
uint8[] navigation_source
float64[] navigation_source_stamp
bool rays_used
string geometry_status
string association_json
string metadata_json
