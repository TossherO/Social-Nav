// Bounded continuous-pose Hybrid A*. No ROS publishers or hardware operations.
// Curves are represented by controls until accepted; rejection never allocates
// dense paths. A reverse 2D wavefront guides the nonholonomic state search.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace {
constexpr double pi=3.14159265358979323846;
using Pose=std::array<double,3>;
using XY=std::array<double,2>;
using Clock=std::chrono::steady_clock;
struct ScopeTime {
  double* elapsed;Clock::time_point start;
  explicit ScopeTime(double* value):elapsed(value),start(value?Clock::now():Clock::time_point{}){}
  ~ScopeTime(){if(elapsed)*elapsed+=std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
};
struct HeuristicCache {
  int width=0,height=0,goal=-1;double resolution=0.,ox=0.,oy=0.;bool valid=false;
  std::vector<int8_t> cells;
  std::vector<uint8_t> costs;
  std::vector<float> potential;
  bool matches(const int8_t* input,const uint8_t* weights,int w,int h,int g,double r,double x,double y)const {
    const size_t n=size_t(w)*h;
    return valid&&width==w&&height==h&&goal==g&&resolution==r&&ox==x&&oy==y&&
      std::memcmp(cells.data(),input,n)==0&&std::memcmp(costs.data(),weights,n)==0;
  }
  void remember(const int8_t* input,const uint8_t* weights,int w,int h,int g,double r,double x,double y) {
    cells.assign(input,input+size_t(w)*h);costs.assign(weights,weights+size_t(w)*h);
    width=w;height=h;goal=g;resolution=r;ox=x;oy=y;valid=true;
  }
};
double wrap(double x) {return x-2*pi*std::floor((x+pi)/(2*pi));}
double mod(double x) {x-=2*pi*std::floor(x/(2*pi));return std::min(x,2*pi-x)<1e-8?0.:x;}
Pose advance(Pose a,double distance,double curvature) {
  const double angle=distance*curvature;
  const double sinc=std::abs(angle)<1e-8?1.-angle*angle/24:std::sin(angle/2)/(angle/2);
  return {a[0]+distance*sinc*std::cos(a[2]+angle/2),a[1]+distance*sinc*std::sin(a[2]+angle/2),a[2]+angle};
}
struct Segment {double distance,curvature;};
// A Dubins connector has at most three segments plus a terminal straight.
struct Curve {std::array<Segment,4> segments;int count=0;double length=0.;bool reverse=false;size_t order=0;};
struct Node {Pose pose;double cost;int parent;Segment edge;};
struct QueueEntry {double score,cost;int index;bool operator<(const QueueEntry& b)const{return score>b.score;}};
struct State {double cost;bool closed=false;};

struct Search {
  const int8_t* cells;const uint8_t* costs;const float* clearance;
  int w,h,vertices,max_states,max_expanded;double res,state_res,ox,oy,radius,margin,budget,reverse_penalty,turn_weight,approach;
  bool reverse,hard_yaw;Pose start,goal;std::vector<XY> footprint;
  double outer=0.;Clock::time_point began=Clock::now();
  int expanded=0,checks=0,joins=0,code=3;double heuristic_ms=0.;
  HeuristicCache* cache;bool cache_enabled=false,profile=false,heuristic_cached=false,wavefront_complete=false;
  double connect_ms=0.,motion_ms=0.,search_loop_ms=0.;
  std::vector<float> private_potential;std::vector<float>& potential;
  explicit Search(HeuristicCache* value=nullptr):cache(value),potential(value?value->potential:private_potential){}
  bool expired()const{return std::chrono::duration<double>(Clock::now()-began).count()>budget;}
  int cell(const Pose& p)const {
    const int x=int(std::floor((p[0]-ox)/res)),y=int(std::floor((p[1]-oy)/res));
    return x<0||y<0||x>=w||y>=h?-1:y*w+x;
  }
  uint64_t key(const Pose& p)const {
    const int bin=(int(std::llround(wrap(p[2])*36/pi))+72)%72;
    const int x=int(std::floor((p[0]-ox)/state_res)),y=int(std::floor((p[1]-oy)/state_res));
    const int columns=int(std::ceil(w*res/state_res));
    return (uint64_t(y)*columns+x)*72+bin;
  }
  bool freePose(const Pose& p,double extra=0.) {
    ++checks;const int centre=cell(p);if(centre<0)return false;
    const double m=margin+extra;
    // EDT is measured between cell centres. Subtract both quantization radii
    // before treating a circle as a sufficient free-space certificate.
    if(clearance[centre]-std::sqrt(2.)*res>outer+m)return true;
    const double c=std::cos(p[2]),s=std::sin(p[2]);
    std::array<XY,16> polygon;double lx=1e100,ly=1e100,hx=-1e100,hy=-1e100;
    for(int i=0;i<vertices;++i) {
      polygon[i]={p[0]+c*footprint[i][0]-s*footprint[i][1],p[1]+s*footprint[i][0]+c*footprint[i][1]};
      lx=std::min(lx,polygon[i][0]);hx=std::max(hx,polygon[i][0]);ly=std::min(ly,polygon[i][1]);hy=std::max(hy,polygon[i][1]);
    }
    const int x0=int(std::floor((lx-m-ox)/res)),x1=int(std::floor((hx+m-ox)/res));
    const int y0=int(std::floor((ly-m-oy)/res)),y1=int(std::floor((hy+m-oy)/res));
    if(x0<0||y0<0||x1>=w||y1>=h)return false;
    // Most poses lie in an entirely free bounding box; do SAT only near edges.
    bool all_free=true;
    for(int y=y0;y<=y1&&all_free;++y)for(int x=x0;x<=x1;++x)if(cells[y*w+x]!=0){all_free=false;break;}
    if(all_free)return true;
    std::array<XY,16> axes;std::array<double,16> low,high;
    for(int i=0;i<vertices;++i) {
      const auto a=polygon[i],b=polygon[(i+1)%vertices];axes[i]={a[1]-b[1],b[0]-a[0]};
      low[i]=1e100;high[i]=-1e100;
      for(int j=0;j<vertices;++j) {const double v=polygon[j][0]*axes[i][0]+polygon[j][1]*axes[i][1];low[i]=std::min(low[i],v);high[i]=std::max(high[i],v);}
    }
    for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x) {
      if(cells[y*w+x]==0)continue;
      const double px=ox+(x+.5)*res,py=oy+(y+.5)*res;
      if(px+res/2+m<lx-1e-10||px-res/2-m>hx+1e-10||py+res/2+m<ly-1e-10||py-res/2-m>hy+1e-10)continue;
      bool hit=true;
      for(int i=0;i<vertices;++i) {
        const double v=px*axes[i][0]+py*axes[i][1],r=(res/2+m)*(std::abs(axes[i][0])+std::abs(axes[i][1]));
        if(v+r<low[i]-1e-10||v-r>high[i]+1e-10){hit=false;break;}
      }
      if(hit)return false;
    }
    return true;
  }
  bool freeMotion(const Pose& from,double distance,double curvature,int depth=0) {
    ScopeTime timer(profile&&depth==0?&motion_ms:nullptr);
    if((checks&255)==0&&expired())return false;
    const Pose mid=advance(from,distance/2,curvature);
    const double enclosure=std::abs(distance)/2*(1.+outer*std::abs(curvature));
    if(freePose(mid,enclosure))return true;
    if(!freePose(mid)||enclosure<1e-4||depth>=15)return false;
    return freeMotion(from,distance/2,curvature,depth+1)&&freeMotion(mid,distance/2,curvature,depth+1);
  }
  bool freeCurve(const Pose& from,const Curve& curve) {
    Pose p=from;
    for(int i=0;i<curve.count;++i) {
      auto segment=curve.segments[i];
      if(!freeMotion(p,segment.distance,segment.curvature))return false;
      p=advance(p,segment.distance,segment.curvature);
    }
    return freePose(p);
  }
  void append(std::vector<Pose>& path,const Curve& curve)const {
    for(int j=0;j<curve.count;++j) {
      auto segment=curve.segments[j];
      const Pose p=path.back();const int n=std::max(1,int(std::ceil(std::max(std::abs(segment.distance)/(res/2),std::abs(segment.distance*segment.curvature)/.05))));
      for(int i=1;i<=n;++i)path.push_back(advance(p,segment.distance*i/n,segment.curvature));
    }
  }
  std::vector<Curve> curves(Pose from,Pose to,bool backwards)const {
    if(backwards){from[2]+=pi;to[2]+=pi;}
    std::vector<Curve> out;
    out.reserve(8);
    auto add=[&](std::initializer_list<Segment> segments) {
      Curve c;c.reverse=backwards;
      for(auto s:segments) {c.length+=s.distance;if(s.distance>1e-9){if(backwards){s.distance=-s.distance;s.curvature=-s.curvature;}c.segments[c.count++]=s;}}
      out.push_back(std::move(c));
    };
    for(double first:{-1.,1.}) {
      XY c1={from[0]-first*radius*std::sin(from[2]),from[1]+first*radius*std::cos(from[2])};
      for(double last:{-1.,1.}) {
        XY c2={to[0]-last*radius*std::sin(to[2]),to[1]+last*radius*std::cos(to[2])};
        const double dx=c2[0]-c1[0],dy=c2[1]-c1[1],d=std::hypot(dx,dy);
        if(d<1e-8) {if(first==last)add({{radius*mod(first*(to[2]-from[2])),first/radius}});continue;}
        const double sine=(last-first)*radius/d;if(std::abs(sine)>1.+1e-10)continue;
        const double tangent=std::atan2(dy,dx)-std::asin(std::max(-1.,std::min(1.,sine)));
        add({{radius*mod(first*(tangent-from[2])),first/radius},
          {std::sqrt(std::max(0.,d*d-(last-first)*(last-first)*radius*radius)),0.},
          {radius*mod(last*(to[2]-tangent)),last/radius}});
      }
      XY c3={to[0]-first*radius*std::sin(to[2]),to[1]+first*radius*std::cos(to[2])};
      const double dx=c3[0]-c1[0],dy=c3[1]-c1[1],d=std::hypot(dx,dy);
      if(d<1e-8||d>4*radius)continue;
      const double height=std::sqrt(std::max(0.,4*radius*radius-d*d/4));
      for(double side:{-1.,1.}) {
        XY c2={(c1[0]+c3[0])/2-side*dy/d*height,(c1[1]+c3[1])/2+side*dx/d*height};
        const double yaw12=std::atan2((c2[1]-c1[1])/2,(c2[0]-c1[0])/2)+first*pi/2;
        const double yaw23=std::atan2((c3[1]-c2[1])/2,(c3[0]-c2[0])/2)-first*pi/2;
        add({{radius*mod(first*(yaw12-from[2])),first/radius},
          {radius*mod(-first*(yaw23-yaw12)),-first/radius},{radius*mod(first*(to[2]-yaw23)),first/radius}});
      }
    }
    return out;
  }
  bool connect(const Pose& p,Curve& chosen) {
    ScopeTime timer(profile?&connect_ms:nullptr);
    ++joins;
    const double distance=std::hypot(goal[0]-p[0],goal[1]-p[1]);
    const double straight=hard_yaw?std::min(approach,distance/2):0.;
    for(int trial=0;trial<(straight>1e-6?2:1);++trial) {
      const double ending=trial==0?straight:0.;Pose anchor=goal;
      if(ending){anchor[0]-=ending*std::cos(goal[2]);anchor[1]-=ending*std::sin(goal[2]);}
      std::vector<Curve> candidates;
      candidates.reserve(reverse?288:144);
      for(int direction=0;direction<(reverse?2:1);++direction) {
        std::vector<double> headings=hard_yaw?std::vector<double>{anchor[2]}:std::vector<double>{std::atan2(anchor[1]-p[1],anchor[0]-p[0]),p[2]};
        if(!hard_yaw)for(int k=0;k<16;++k)headings.push_back(k*pi/8);
        for(double heading:headings) {
          anchor[2]=heading;
          for(auto curve:curves(p,anchor,direction==1)) {
            // A reversing connector must not acquire a forward terminal segment.
            if(ending&&direction==1)continue;
            if(ending){curve.segments[curve.count++]={ending,0.};curve.length+=ending;}
            curve.order=candidates.size();
            candidates.push_back(std::move(curve));
          }
        }
      }
      // Only the first twelve are checked. Explicit insertion order retains
      // stable-sort ties while avoiding sorting the discarded candidates.
      std::partial_sort(candidates.begin(),candidates.begin()+std::min(size_t(12),candidates.size()),candidates.end(),
        [&](const Curve& a,const Curve& b){
          const double ca=a.length*(a.reverse?reverse_penalty:1.),cb=b.length*(b.reverse?reverse_penalty:1.);
          return ca==cb?a.order<b.order:ca<cb;
        });
      int tried=0;
      for(const auto& curve:candidates) {
        if(expired()||tried++>=12)break;
        if(freeCurve(p,curve)){chosen=curve;return true;}
      }
    }
    return false;
  }
  void wavefront() {
    const auto t=Clock::now();const int g=cell(goal);
    if(cache_enabled&&cache&&cache->matches(cells,costs,w,h,g,res,ox,oy)) {
      heuristic_cached=true;wavefront_complete=true;
      heuristic_ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();return;
    }
    if(cache)cache->valid=false;
    potential.assign(w*h,std::numeric_limits<float>::infinity());
    using Entry=std::pair<float,int>;std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> q;
    potential[g]=0.;q.push({0.,g});
    const int dx[8]={1,-1,0,0,1,1,-1,-1},dy[8]={0,0,1,-1,1,-1,1,-1};
    int pops=0;
    while(!q.empty()) {
      if((++pops&1023)==0&&expired())break;
      auto entry=q.top();q.pop();const int i=entry.second;if(entry.first!=potential[i])continue;
      const int x=i%w,y=i/w;
      for(int k=0;k<8;++k) {
        const int nx=x+dx[k],ny=y+dy[k];if(nx<0||ny<0||nx>=w||ny>=h)continue;
        const int j=ny*w+nx;if(cells[j]!=0)continue;
        const float cost=entry.first+res*(k<4?1.:std::sqrt(2.))*(1.+.25*costs[j]/254.);
        if(cost<potential[j]){potential[j]=cost;q.push({cost,j});}
      }
    }
    wavefront_complete=q.empty();
    if(wavefront_complete&&cache_enabled&&cache)cache->remember(cells,costs,w,h,g,res,ox,oy);
    heuristic_ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
  }
  double heuristic(const Pose& p)const {
    const int i=cell(p);const double d=std::hypot(p[0]-goal[0],p[1]-goal[1]);
    double estimate=i>=0&&std::isfinite(potential[i])?std::max(d,double(potential[i])):d;
    if(hard_yaw&&!reverse) {
      double dubins=std::numeric_limits<double>::infinity();
      for(const auto& curve:curves(p,goal,false))dubins=std::min(dubins,curve.length);
      estimate=std::max(estimate,dubins);
    }
    return estimate;
  }
  std::vector<Pose> solve(int& visited) {
    if(!freePose(start)){code=4;return {};}
    Curve ending;
    std::vector<Pose> incumbent;double incumbent_cost=std::numeric_limits<double>::infinity();
    auto connectionCost=[&](const Curve& curve) {
      double value=0.;for(int i=0;i<curve.count;++i){auto edge=curve.segments[i];value+=std::abs(edge.distance)*(1.+turn_weight*radius*std::abs(edge.curvature))*(curve.reverse?reverse_penalty:1.);}
      return value;
    };
    if(connect(start,ending)) {
      incumbent={start};append(incumbent,ending);incumbent_cost=connectionCost(ending);
      // A collision-free Dubins word is not necessarily a good obstacle route.
      // Retain long analytic detours as an incumbent while searching multi-bend
      // alternatives; return immediately only near the obstacle-free lower bound.
      double lower=std::hypot(start[0]-goal[0],start[1]-goal[1]);
      if(hard_yaw&&!reverse) {
        lower=std::numeric_limits<double>::infinity();
        for(const auto& curve:curves(start,goal,false))lower=std::min(lower,curve.length);
      }
      if(ending.length<=1.1*lower+1e-7){code=0;visited=1;return incumbent;}
    }
    wavefront();if(expired()){code=incumbent.empty()?1:0;return incumbent;}
    // An occupied/unknown barrier separating the base centre is a proof only
    // after a complete wavefront, and only when that centre lies inside the
    // footprint. A partial distance field must never become a no-path verdict.
    bool positive=true,negative=true;
    for(int i=0;i<vertices;++i) {
      const auto a=footprint[i],b=footprint[(i+1)%vertices];
      const double cross=a[0]*b[1]-a[1]*b[0];
      positive&=cross>0.;negative&=cross<0.;
    }
    if(wavefront_complete&&(positive||negative)&&!std::isfinite(potential[cell(start)])&&incumbent.empty()) {
      code=5;return {};
    }
    ScopeTime loop_timer(profile?&search_loop_ms:nullptr);
    const double step=std::max(2*res,std::min(.35,radius*pi/12));
    std::vector<Node> nodes;nodes.reserve(std::min(max_states,50000));
    std::unordered_map<uint64_t,State> states;states.reserve(std::min(max_states,100000));
    std::priority_queue<QueueEntry> queue;
    nodes.push_back({start,0.,-1,{0.,0.}});states[key(start)]={0.,false};queue.push({heuristic(start),0.,0});
    while(!queue.empty()&&expanded<max_expanded&&int(states.size())<max_states) {
      if(expired()){code=1;break;}
      auto entry=queue.top();queue.pop();Node node=nodes[entry.index];
      if(entry.score>=incumbent_cost)break;
      auto& state=states.find(key(node.pose))->second;
      if(entry.cost>state.cost+1e-9)continue;
      // One expansion per discretized state. Weighted Hybrid A* is bounded and
      // approximate; repeatedly refining almost identical poses wastes its
      // budget without improving the independent path certificate.
      if(state.closed)continue;
      state.closed=true;
      ++expanded;
      if(expanded%16==0||std::hypot(node.pose[0]-goal[0],node.pose[1]-goal[1])<step) {
        if(connect(node.pose,ending)) {
          const double value=node.cost+connectionCost(ending);
          if(value<incumbent_cost) {
            std::vector<int> chain;int root=entry.index;
            for(;nodes[root].parent>=0;root=nodes[root].parent)chain.push_back(root);
            incumbent={nodes[root].pose};
            for(auto i=chain.rbegin();i!=chain.rend();++i) {
              Curve edge;edge.segments[0]=nodes[*i].edge;edge.count=1;append(incumbent,edge);
            }
            append(incumbent,ending);incumbent_cost=value;
          }
        }
      }
      for(int direction=0;direction<(reverse?2:1);++direction)for(double turn:{-1.,-.5,0.,.5,1.}) {
        Segment edge{direction==0?step:-step,turn/radius};Pose p=advance(node.pose,edge.distance,edge.curvature);
        const int c=cell(p);if(c<0||cells[c]!=0)continue;
        const double cost=node.cost+step*(1.+turn_weight*std::abs(turn)+.25*costs[c]/254.)*(direction==1?reverse_penalty:1.);
        const auto id=key(p);
        auto old=states.find(id);if(old!=states.end()&&(old->second.closed||cost>=old->second.cost))continue;
        if(!freeMotion(node.pose,edge.distance,edge.curvature))continue;
        states[id]={cost,false};nodes.push_back({p,cost,entry.index,edge});queue.push({cost+1.2*heuristic(p),cost,int(nodes.size()-1)});
      }
    }
    visited=int(states.size());if(!incumbent.empty()){code=0;return incumbent;}
    if(code!=1)code=queue.empty()?3:2;return {};
  }
};
}

// params: radius, margin, max_states, max_expanded, seconds, reverse,
// reverse_penalty, turn_weight, preferred_final_straight, state_resolution_m.
// stats: elapsed_ms, expanded, visited, collision_checks, heuristic_ms, joins, reason.
// The cached entry appends connector, motion, loop timings and a cache-hit flag.
// Timings are optional and nested: connector and motion intervals can overlap.
static int plan_search(HeuristicCache* cache,int options,const int8_t* cells,const uint8_t* costs,const float* clearance,
    int width,int height,double resolution,const double* origin,const double* polygon,int vertices,
    const double* start,const double* goal,int hard_yaw,const double* params,
    double* output,int capacity,double* stats,int stats_size) noexcept {
  try {
    if(!cells||!costs||!clearance||!origin||!polygon||!start||!goal||!params||!output||!stats||
        width<1||height<1||int64_t(width)*height>4000000||vertices<3||vertices>16||capacity<2||
        !std::isfinite(resolution)||resolution<=0.)return -1;
    for(int i=0;i<10;++i)if(!std::isfinite(params[i]))return -1;
    for(int i=0;i<3;++i)if(!std::isfinite(start[i])||!std::isfinite(goal[i]))return -1;
    for(int i=0;i<2*vertices;++i)if(!std::isfinite(polygon[i]))return -1;
    if(params[0]<=0||params[1]<0||params[2]<1||params[2]>600000||params[3]<1||params[4]<=0||params[4]>5||params[9]<=0||params[9]>1)return -1;
    Search s(cache);s.cache_enabled=options&1;s.profile=options&2;
    s.cells=cells;s.costs=costs;s.clearance=clearance;s.w=width;s.h=height;s.res=resolution;s.ox=origin[0];s.oy=origin[1];s.vertices=vertices;
    s.radius=params[0];s.margin=params[1];s.max_states=int(params[2]);s.max_expanded=int(params[3]);s.budget=params[4];s.reverse=params[5]!=0;
    s.reverse_penalty=params[6];s.turn_weight=params[7];s.approach=params[8];s.hard_yaw=hard_yaw!=0;
    s.state_res=std::max(resolution,params[9]);
    s.start={start[0],start[1],start[2]};s.goal={goal[0],goal[1],goal[2]};
    if(s.cell(s.start)<0||s.cell(s.goal)<0)return -1;
    for(int i=0;i<vertices;++i){s.footprint.push_back({polygon[2*i],polygon[2*i+1]});s.outer=std::max(s.outer,std::hypot(polygon[2*i],polygon[2*i+1]));}
    int visited=0;auto result=s.solve(visited);
    const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-s.began).count();
    const double values[11]={elapsed,double(s.expanded),double(visited),double(s.checks),s.heuristic_ms,
      double(s.joins),double(s.code),s.connect_ms,s.motion_ms,s.search_loop_ms,double(s.heuristic_cached)};
    std::copy(values,values+stats_size,stats);
    if(int(result.size())>capacity)return -2;
    for(size_t i=0;i<result.size();++i)for(int j=0;j<3;++j)output[3*i+j]=result[i][j];
    return int(result.size());
  }catch(...){return -3;}
}

extern "C" void* social_nav_pose_create() noexcept {
  try{return new HeuristicCache;}catch(...){return nullptr;}
}
extern "C" void social_nav_pose_destroy(void* context) noexcept {
  delete static_cast<HeuristicCache*>(context);
}
extern "C" int social_nav_pose_plan_cached(void* context,const int8_t* cells,const uint8_t* costs,const float* clearance,
    int width,int height,double resolution,const double* origin,const double* polygon,int vertices,
    const double* start,const double* goal,int hard_yaw,const double* params,
    double* output,int capacity,double* stats,int options) noexcept {
  if(!context)return -1;
  return plan_search(static_cast<HeuristicCache*>(context),options,cells,costs,clearance,width,height,resolution,
    origin,polygon,vertices,start,goal,hard_yaw,params,output,capacity,stats,11);
}
extern "C" int social_nav_pose_plan(const int8_t* cells,const uint8_t* costs,const float* clearance,
    int width,int height,double resolution,const double* origin,const double* polygon,int vertices,
    const double* start,const double* goal,int hard_yaw,const double* params,
    double* output,int capacity,double* stats) noexcept {
  return plan_search(nullptr,0,cells,costs,clearance,width,height,resolution,origin,polygon,vertices,
    start,goal,hard_yaw,params,output,capacity,stats,7);
}
