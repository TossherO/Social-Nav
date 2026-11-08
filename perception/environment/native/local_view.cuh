// Resident CUDA local history and exact squared-Euclidean distance transforms.
// Included inside occupancy_cuda.cu's private namespace; no ROS/Python dependency.
// View channels: original six, ground, support, clearance, unknown distance,
// frontier, historical free, classification time, recent obstacle, hit, expiry.
__global__ void view_base(const double* raw,const double* floor,const uint8_t* support,int size,double* out,bool suspended) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;
    for(int k=0;k<6;++k)out[k*size+i]=raw[k*size+i];
    out[6*size+i]=float(floor[i]);out[7*size+i]=support[i]==1;
    if(suspended || support[i]==2) {if(out[i]==0)out[i]=-1;out[7*size+i]=0;}
    for(int k=8;k<16;++k)out[k*size+i]=(k==10 || k==11 || k==13)?0.:NAN;
}
__global__ void scan_ring(NearPolicy p, double* ring) {
    __shared__ double angles[360], radii[360], height;
    __shared__ int first;
    int j = threadIdx.x;
    double* info = ring + 724;

    // All threads use the same validity result before computing intersections.
    if (j == 0) {
        for (int k = 0; k < 4; ++k)
            info[k] = 0.;
        if (p.has_plane) {
            if (acos(fmin(1., fmax(-1., p.pose[10]))) > p.settings[4])
                info[0] = 2.;
            else {
                height = p.pose[11] - (p.plane[0] * p.pose[3] + p.plane[1] * p.pose[7] + p.plane[2]);
                info[0] = !isfinite(height) || height <= 0 ? 3. : 1.;
            }
        }
    }
    __syncthreads();
    if (info[0] != 1.)
        return;

    if (j < 360) {
        double beta = p.settings[1] * 3.14159265358979323846 / 180., cb = cos(beta), sb = sin(beta);
        double phi = j * 3.14159265358979323846 / 180.;
        double u = cos(phi) * cb, v = sin(phi) * cb;
        double x = p.pose[0] * u + p.pose[1] * v + p.pose[2] * sb;
        double y = p.pose[4] * u + p.pose[5] * v + p.pose[6] * sb;
        double z = p.pose[8] * u + p.pose[9] * v + p.pose[10] * sb;
        double denominator = z - (x * p.plane[0] + y * p.plane[1]);
        double distance = denominator < -1e-6 ? height / -denominator * sqrt(x * x + y * y) : 0.;
        radii[j] = fmax(0., fmin(p.settings[2], distance) - p.settings[3]);
        angles[j] = atan2(y, x);
    }
    __syncthreads();

    // Preserve the serial tie order when locating the angle wrap point.
    if (j == 0) {
        first = 0;
        double minimum = INFINITY, maximum = 0.;
        for (int k = 0; k < 360; ++k) {
            if (angles[k] < angles[first])
                first = k;
            minimum = fmin(minimum, radii[k]);
            maximum = fmax(maximum, radii[k]);
        }
        info[1] = height;
        info[2] = minimum;
        info[3] = maximum;
    }
    __syncthreads();

    if (j < 360) {
        int source = (first + j) % 360;
        ring[j + 1] = angles[source];
        ring[362 + j + 1] = radii[source];
    }
    __syncthreads();

    if (j == 0) {
        ring[0] = ring[360] - 2 * 3.14159265358979323846;
        ring[361] = ring[1] + 2 * 3.14159265358979323846;
        ring[362] = ring[722];
        ring[723] = ring[363];
    }
}

__device__ bool in_ring(int x,int y,NearPolicy p,const double* ring) {
    if(ring[724]!=1)return false;
    double cx=(x+.5)*p.resolution+p.x-p.pose[3],cy=(y+.5)*p.resolution+p.y-p.pose[7];
    if(fabs(cx)>p.settings[2] || fabs(cy)>p.settings[2])return false;
    for(int k=0;k<4;++k) {
        double u=cx+((k&1)? .5:-.5)*p.resolution,v=cy+((k&2)? .5:-.5)*p.resolution;
        double a=atan2(v,u);int low=0,high=361;
        while(high-low>1) { int mid=(low+high)/2;if(ring[mid]<=a)low=mid;else high=mid; }
        double fraction=(a-ring[low])/(ring[high]-ring[low]);
        double radius=ring[362+low]+fraction*(ring[362+high]-ring[362+low]);
        if(sqrt(u*u+v*v)>radius)return false;
    }
    return true;
}
__global__ void memory_prepare(const NearCell* before,NearCell* after,unsigned* flags,double* hit_bounds,
    const double* floor,const uint8_t* support,const double* raw,NearPolicy p,const double* ring,int dx,int dy,bool initialized) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,size=p.height*p.width;if(i>=size)return;
    int x=i%p.width,y=i/p.width,ox=x-dx,oy=y-dy;
    NearCell c{NAN,NAN,NAN,NAN,INFINITY,-INFINITY,NAN,0,NAN};
    if(initialized && ox>=0 && ox<p.width && oy>=0 && oy<p.height)c=before[oy*p.width+ox];
    bool paused=p.history_suspended || support[i]==2;
    bool valid=isfinite(c.classified),ground=!paused && support[i]==1 && isfinite(floor[i]);
    bool conflict=valid && ground && fabs(floor[i]-c.free_ground)>p.settings[5];
    if(ground)c.ground=float(floor[i]);after[i]=c;
    flags[i]=(raw[i]==100?1:0)|(conflict?8:0)|(in_ring(x,y,p,ring)?16:0)|(paused?32:0);
    hit_bounds[i]=INFINITY;hit_bounds[size+i]=-INFINITY;
}
__global__ void memory_hits(const double* xyz,int n,const uint8_t* moving,NearCell* cells,unsigned* flags,
    double* hit_bounds,NearPolicy p) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    int x=floor((xyz[3*j]-p.x)/p.resolution),y=floor((xyz[3*j+1]-p.y)/p.resolution);
    if(x<0 || x>=p.width || y<0 || y>=p.height)return;
    int i=y*p.width+x,size=p.height*p.width;const auto& c=cells[i];
    double ground=p.settings[7]?c.ground:c.free_ground,h=xyz[3*j+2]-ground;
    if((!p.settings[7] && !isfinite(c.classified)) || !isfinite(h) || h<p.minimum || h>p.maximum)return;
    atomicOr(flags+i,2u|((!moving || !moving[j])?4u:0u));
    if(p.settings[7]) {min_atomic(hit_bounds+i,xyz[3*j+2]);max_atomic(hit_bounds+size+i,xyz[3*j+2]);}
}
__global__ void memory_finish(NearCell* cells,const unsigned* flags,const double* hit_bounds,NearPolicy p,
    double* out,int* counts) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,size=p.height*p.width;if(i>=size)return;
    auto& c=cells[i];unsigned f=flags[i];bool hit=f&3,conflict=f&8,blind=f&16;
    bool paused=f&32;
    bool valid=isfinite(c.classified),certified=!paused && out[i]==0 && out[7*size+i] && out[3*size+i]>=p.coverage && !hit;
    if(valid && hit)atomicAdd(counts+3,1);
    if(valid && hit && !p.settings[7])atomicAdd(counts+1,1);
    if(conflict)atomicAdd(counts+2,1);
    if(p.settings[7] && (f&2)) {
        if(f&4)c.ordinary_stamp=p.stamp;
        c.hit_stamp=p.stamp;c.expires=p.stamp+((f&4)?p.settings[8]:fmin(p.settings[8],p.settings[9]));
        c.expires=fmax(c.expires,c.ordinary_stamp+p.settings[8]);
        c.hit_min=hit_bounds[i];c.hit_max=hit_bounds[size+i];
    }
    valid=((valid && (p.settings[6] || blind)) || certified) && !(hit && !p.settings[7]) && !conflict;
    if(!valid)c.classified=NAN;
    if(certified && !conflict) {
        c.classified=p.stamp;c.free_ground=float(out[6*size+i]);c.free_coverage=float(out[3*size+i]);c.free_seen=out[2*size+i];
        c.hit_stamp=NAN;c.expires=NAN;c.ordinary_stamp=-INFINITY;
    }
    bool active=isfinite(c.ground) && blind && isfinite(c.hit_stamp) && p.stamp<=c.expires;
    bool restored=!paused && valid && blind && out[i]==-1 && !active;
    if(valid)atomicAdd(counts,1);
    if(restored) {
        atomicAdd(counts+4,1);out[i]=0;out[size+i]=1;out[2*size+i]=c.free_seen;out[3*size+i]=c.free_coverage;
        out[4*size+i]=NAN;out[5*size+i]=NAN;out[6*size+i]=c.free_ground;out[7*size+i]=1;
        out[11*size+i]=1;out[12*size+i]=c.classified;
    }
    if(active) {
        atomicAdd(counts+5,1);out[i]=100;out[size+i]=1;out[2*size+i]=fmax(out[2*size+i],c.hit_stamp);
        out[3*size+i]=0;out[4*size+i]=float(c.hit_min-c.ground);out[5*size+i]=float(c.hit_max-c.ground);
        out[6*size+i]=c.ground;out[7*size+i]=1;out[13*size+i]=1;
    }
    if(blind) {out[14*size+i]=c.hit_stamp;out[15*size+i]=c.expires;}
}
// Separable exact EDT. Integer squared distances avoid jump-flood approximation.
// Each pixel scans one row/column; bounded local grids keep this inexpensive.
__global__ void distance_rows(const double* grid,int h,int w,float* rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=h*w)return;
    int x=i%w,y=i/w;float occupied=INFINITY,unknown=fminf(float((x+1)*(x+1)),float((w-x)*(w-x)));
    for(int j=0;j<w;++j) {
        float d=(x-j)*(x-j);double v=grid[y*w+j];
        if(v==100)occupied=fminf(occupied,d);if(v==-1)unknown=fminf(unknown,d);
    }
    rows[i]=occupied;rows[h*w+i]=unknown;
}
__global__ void distance_columns(double* grid,int h,int w,double resolution,const float* rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,size=h*w;if(i>=size)return;
    int x=i%w,y=i/w;float occupied=INFINITY,unknown=fminf(float((y+1)*(y+1)),float((h-y)*(h-y)));
    for(int j=0;j<h;++j) {
        float d=(y-j)*(y-j);occupied=fminf(occupied,rows[j*w+x]+d);unknown=fminf(unknown,rows[size+j*w+x]+d);
    }
    grid[8*size+i]=grid[i]==-1 || !isfinite(occupied)?NAN:sqrtf(occupied)*float(resolution);
    grid[9*size+i]=sqrtf(unknown)*float(resolution);
    grid[10*size+i]=(x>0 && y>0 && x<w-1 && y<h-1 && grid[i]==0 &&
        (grid[i-1]==-1 || grid[i+1]==-1 || grid[i-w]==-1 || grid[i+w]==-1))?1.:0.;
}
__global__ void person_background(const unsigned* tags,const NearCell* memory,const uint8_t* support,int size,double* view,bool suspended) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;
    view[16*size+i]=!suspended && support[i]!=2 && view[i]==100 && tags[i]==1 && isfinite(memory[i].classified)?1.:0.;
}
void local_view(Tree& t,int h,int w,const double* origin,const double* params,double stamp,const double* pose,
    const double* plane,const double* settings,const uint8_t* moving,int n,double* output,int* counts,double* ring_info,bool people=false,bool device_output=false) {
    int size=h*w,channels=people?17:16;
    if(!device_output)t.view.reserve(size_t(size)*channels);
    double* view=device_output?output:t.view.data;
    t.distance_rows.reserve(size_t(size)*2);
    bool suspended=settings && settings[0] && t.history_suspended;
    view_base<<<blocks(size),256>>>(t.output.data,t.floor.data,t.support.data,size,view,suspended);
    int host_counts[6]={};double host_ring[4]={};
    if(settings && settings[0]) {
        NearPolicy p{};p.height=h;p.width=w;p.x=origin[0];p.y=origin[1];p.resolution=params[0];p.stamp=stamp;
        p.minimum=params[1];p.maximum=params[2];p.coverage=params[5];p.has_plane=plane!=nullptr;
        p.history_suspended=suspended;
        std::copy(settings,settings+10,p.settings);std::copy(pose,pose+16,p.pose);
        if(plane)std::copy(plane,plane+3,p.plane);
        bool compatible=t.memory_initialized && t.memory_height==h && t.memory_width==w &&
            t.memory_resolution==params[0] && stamp>t.memory_stamp;
        double sx=(t.memory_x-origin[0])/params[0],sy=(t.memory_y-origin[1])/params[0];
        if(compatible && (fabs(sx-round(sx))>1e-6 || fabs(sy-round(sy))>1e-6))
            throw std::runtime_error("Near-field memory grid moved off its source lattice");
        t.memory_a.reserve(size);t.memory_b.reserve(size);t.memory_flags.reserve(size);t.hit_bounds.reserve(size_t(size)*2);
        t.ring.reserve(1448);t.memory_counts.reserve(6);check(cudaMemset(t.memory_counts.data,0,6*sizeof(int)));
        if(moving)upload(t.view_transient,moving,n);
        scan_ring<<<1,512>>>(p,t.ring.data);
        memory_prepare<<<blocks(size),256>>>(t.memory_a.data,t.memory_b.data,t.memory_flags.data,t.hit_bounds.data,
            t.floor.data,t.support.data,t.output.data,p,t.ring.data,int(round(sx)),int(round(sy)),compatible);
        if(n)memory_hits<<<blocks(n),256>>>(t.endpoints.data,n,moving?t.view_transient.data:nullptr,t.memory_b.data,
            t.memory_flags.data,t.hit_bounds.data,p);
        memory_finish<<<blocks(size),256>>>(t.memory_b.data,t.memory_flags.data,t.hit_bounds.data,p,view,t.memory_counts.data);
        std::swap(t.memory_a.data,t.memory_b.data);std::swap(t.memory_a.capacity,t.memory_b.capacity);
        t.memory_initialized=true;t.memory_height=h;t.memory_width=w;t.memory_x=origin[0];t.memory_y=origin[1];
        t.memory_resolution=params[0];t.memory_stamp=stamp;
        if(people) {
            t.person_columns.reserve(size);check(cudaMemset(t.person_columns.data,0,size*sizeof(unsigned)));
            Projection q{h,w,0,origin[0],origin[1],params[0],params[1],params[2],params[3],params[4],params[5],params[6]!=0};
            if(t.initialized)person_voxels<<<blocks(t.grid.dim.x*t.grid.dim.y*t.grid.dim.z),256>>>(t.voxels.data,t.grid,t.floor.data,q,t.person_columns.data,stamp,t.obstacle_ttl);
            if(n)person_endpoints<<<blocks(n),256>>>(t.endpoints.data,n,moving?t.view_transient.data:nullptr,t.floor.data,q,t.person_columns.data);
            person_background<<<blocks(size),256>>>(t.person_columns.data,t.memory_a.data,t.support.data,size,view,suspended);
        }
        check(cudaMemcpy(host_counts,t.memory_counts.data,6*sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_ring,t.ring.data+724,4*sizeof(double),cudaMemcpyDeviceToHost));
    } else if(people)check(cudaMemset(view+16*size,0,size*sizeof(double)));
    distance_rows<<<blocks(size),256>>>(view,h,w,t.distance_rows.data);
    distance_columns<<<blocks(size),256>>>(view,h,w,params[0],t.distance_rows.data);
    check(cudaGetLastError());
    if(!device_output)check(cudaMemcpy(output,view,size_t(size)*channels*sizeof(double),cudaMemcpyDeviceToHost));
    std::copy(host_counts,host_counts+6,counts);std::copy(host_ring,host_ring+4,ring_info);
}
