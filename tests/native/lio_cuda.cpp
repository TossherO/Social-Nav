// Standalone CUDA test; no bag, ROS, Torch or model weights are needed.
#include "cuda_matcher.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void require(bool good,const char* why){if(!good)throw std::runtime_error(why);}
bool inside(const LioPoint& p,const LioBox& b){return p.x>=b.low[0]&&p.x<b.high[0]&&p.y>=b.low[1]&&p.y<b.high[1]&&p.z>=b.low[2]&&p.z<b.high[2];}
float distance(const LioPoint& a,const LioPoint& b){float x=a.x-b.x,y=a.y-b.y,z=a.z-b.z;return x*x+y*y+z*z;}

int main(){try{
    std::mt19937 random(743);std::uniform_real_distribution<float> value(-5,5);
    std::vector<LioPoint> map,queries;
    for(int i=0;i<12000;++i)map.push_back({value(random),value(random),value(random),float(i)});
    for(int i=0;i<200;++i)queries.push_back({value(random),value(random),value(random),0});
    LioCuda matcher(0,.2f,false);matcher.search_resolution(.25f);matcher.reset(map.data(),map.size());
    auto verify=[&](){
        matcher.sync_map();require(matcher.stats().points==map.size(),"map mutation count differs");
        matcher.begin_scan(queries.data(),queries.size());std::vector<LioMatch> out(queries.size());
        matcher.match(queries.data(),queries.size(),true,out.data());
        for(size_t i=0;i<queries.size();++i){
            std::vector<std::pair<float,float>> ordered;
            for(auto p:map)ordered.push_back({distance(p,queries[i]),p.intensity});
            std::sort(ordered.begin(),ordered.end());
            if(ordered[4].first>5)continue;require(out[i].count==5,"missing nearest points");
            for(int j=0;j<5;++j){require(std::abs(out[i].distance[j]-ordered[j].first)<1e-6,"wrong nearest distance");
                require(out[i].neighbors[j].intensity==ordered[j].second,"wrong nearest identity");}
        }
        // Rejected observations must stay rejected when the filter reuses NN.
        for(auto& m:out){m.selected=0;m.flags=0;}
        std::vector<int> ids(out.size());for(size_t i=0;i<ids.size();++i)ids[i]=i;
        matcher.correct(ids.data(),out.data(),out.size());matcher.match(queries.data(),queries.size(),false,out.data());
        for(auto& m:out)require(m.selected==0&&m.flags==0,"cached rejected observation revived");
        std::vector<uint8_t> selected(out.size(),0);selected[0]=1;
        matcher.select(selected.data(),selected.size());matcher.match(queries.data(),queries.size(),false,out.data());
        require(out[0].flags==2,"host QR selection was not preserved");
    };
    verify();
    for(int round=0;round<3;++round){
        for(int i=0;i<200;++i){auto p=LioPoint{value(random),value(random),value(random),float(20000+round*200+i)};
            LioBox box{};const float v[3]={p.x,p.y,p.z};
            for(int j=0;j<3;++j){box.low[j]=std::floor(v[j]/.2f)*.2f;box.high[j]=box.low[j]+.2f;}
            map.erase(std::remove_if(map.begin(),map.end(),[&](LioPoint x){return inside(x,box);}),map.end());
            matcher.insert(p,&box);map.push_back(p);
            if(i%4==0){matcher.insert(p,&box);map.erase(std::remove_if(map.begin(),map.end(),[&](LioPoint x){return inside(x,box);}),map.end());map.push_back(p);}
        }
        LioBox erase{};for(int j=0;j<3;++j){erase.low[j]=-10;erase.high[j]=10;}erase.high[round]=-3;
        matcher.erase(erase);map.erase(std::remove_if(map.begin(),map.end(),[&](LioPoint x){return inside(x,erase);}),map.end());
        LioPoint p{-4,-4,-4,31000.f+round};matcher.insert(p,nullptr);map.push_back(p);verify();
    }
    // Duplicates must be retained; ties request the original CPU ordering.
    LioPoint p{0,0,0,0};map.assign(8,p);matcher.reset(map.data(),map.size());matcher.begin_scan(&p,1);
    LioMatch m;matcher.match(&p,1,true,&m);require(m.flags&1,"nearest tie did not request reference ordering");
    matcher.reset(nullptr,0);matcher.begin_scan(&p,1);matcher.match(&p,1,true,&m);
    require(m.flags&1&&m.count==0&&!m.selected&&matcher.stats().points==0,"empty map reused stale index");
    // A well-conditioned plane is solved on device; collinear points defer QR.
    LioCuda fit(0,.2f,true);map.clear();
    for(int i=-2;i<=2;++i)for(int j=-2;j<=2;++j)map.push_back({i*.2f,j*.2f,1,0});
    p={.073f,.091f,1.01f,0};fit.reset(map.data(),map.size());fit.begin_scan(&p,1);fit.match(&p,1,true,&m);
    require(!m.flags&&m.selected,"well-conditioned plane rejected");
    require(std::abs(m.normal[2]+1)<1e-5&&std::abs(m.normal[3]+.01f)<1e-5,"wrong plane residual");
    map.clear();for(int i=-3;i<=3;++i)map.push_back({i*.1f,0,1,0});
    fit.reset(map.data(),map.size());fit.begin_scan(&p,1);fit.match(&p,1,true,&m);require(m.flags,"degenerate plane did not request CPU QR");
    LioImuPose imu[2]{};LioDeskewState state{};
    for(int j=0;j<3;++j){imu[0].rotation[j*3+j]=imu[1].rotation[j*3+j]=1;
        state.extrinsic[j*3+j]=state.inverse_extrinsic[j*3+j]=state.inverse_rotation[j*3+j]=1;}
    imu[1].offset=.1;imu[0].velocity[0]=1;state.position[0]=.1;
    LioTimedPoint timed[3]={{1,2,3,9,0},{1,2,3,9,50},{1,2,3,9,100}};
    fit.deskew(timed,3,imu,2,state);
    require(timed[0].x==1&&timed[0].intensity==9&&std::abs(timed[1].x-.95f)<1e-7&&timed[2].x==1,"wrong translational deskew or zero-time behavior");
    imu[0].velocity[0]=0;state.position[0]=0;imu[1].angular_velocity[2]=1;
    state.inverse_rotation[0]=state.inverse_rotation[4]=std::cos(.1);state.inverse_rotation[1]=std::sin(.1);state.inverse_rotation[3]=-std::sin(.1);
    LioTimedPoint rotating{1,0,0,4,50};fit.deskew(&rotating,1,imu,2,state);
    require(std::abs(rotating.x-std::cos(.05))<1e-7&&std::abs(rotating.y+std::sin(.05))<1e-7,"wrong rotational deskew");
    LioImuPose carried[3]{};
    for(auto& p:carried){for(int j=0;j<3;++j)p.rotation[3*j+j]=1;p.velocity[0]=1;}
    carried[1].offset=-.01;carried[2].offset=.1;LioDeskewState boundary{};
    for(int j=0;j<3;++j)boundary.extrinsic[3*j+j]=boundary.inverse_extrinsic[3*j+j]=boundary.inverse_rotation[3*j+j]=1;
    boundary.position[0]=.1;LioTimedPoint first[2]={{1,0,0,0,1},{1,0,0,0,1}};
    fit.deskew(first,2,carried,3,boundary);
    require(std::abs(first[0].x-.812f)<2e-7&&std::abs(first[1].x-.911f)<2e-7,"carried IMU anchor or first-point semantics differ");
    LioPoint body[4]={{1,2,3,0},{2,1,4,0},{-2,-1,0,0},{4,5,6,0}},normals[4];uint8_t selected[4]={1,0,1,1};
    for(auto& n:normals)n={0,0,1,-.1f};
    fit.begin_scan(body,4);fit.select(selected,4);double linear_state[21]{};
    for(int j=0;j<3;++j)linear_state[j*3+j]=linear_state[12+j*3+j]=1;
    linear_state[9]=.2;linear_state[10]=.1;linear_state[11]=-.3;
    for(bool estimate:{false,true}){double actual[144],rhs[12],expected[144]{},expected_rhs[12]{};
        fit.linearize(normals,4,linear_state,estimate,actual,rhs);
        for(int i=0;i<4;++i)if(selected[i]){double h[12]={0,0,1,body[i].y+.1,-body[i].x-.2,0,0,0,0,0,0,0};
            if(estimate){h[6]=body[i].y;h[7]=-body[i].x;h[11]=1;}
            for(int r=0;r<12;++r){expected_rhs[r]+=h[r]*double(.1f);for(int c=0;c<12;++c)expected[12*r+c]+=h[r]*h[c];}}
        for(int r=0;r<12;++r){require(std::abs(rhs[r]-expected_rhs[r])<1e-10,"wrong observation gradient");
            for(int c=0;c<12;++c)require(std::abs(actual[12*r+c]-expected[12*r+c])<1e-10,"wrong observation Hessian");}}
    LioPoint one_body{1,2,3,0},one_normal{0,1,0,-.25f};uint8_t one_selected=1;
    fit.begin_scan(&one_body,1);fit.select(&one_selected,1);
    double rotated_state[21]={0,-1,0,1,0,0,0,0,1,.2,.1,-.3,1,0,0,0,0,-1,0,1,0};
    double expected_h[12]={0,1,0,1.1,1.8,0,2,-1,0,0,0,1},actual_hth[144],actual_htr[12];
    fit.linearize(&one_normal,1,rotated_state,true,actual_hth,actual_htr);
    for(int r=0;r<12;++r){require(std::abs(actual_htr[r]-expected_h[r]*.25)<1e-10,"rotated observation gradient mismatch");
        for(int c=0;c<12;++c)require(std::abs(actual_hth[r*12+c]-expected_h[r]*expected_h[c])<1e-10,"rotated observation Hessian mismatch");}
    std::cout<<"CUDA map/NN, cache/plane, deskew boundaries and observation reduction checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
