#include "ground_pick.h"
#include <iostream>
#include <stdexcept>
using namespace social_nav_rviz;
void check(bool condition) { if (!condition) throw std::runtime_error("Ground picking check failed"); }
int main() {
  Point hit,flat;
  const Point origin{5.,0.,5.},ray{-.5,0.,-1.};
  const std::vector<Point> ground{{1.25,0.,-2.5}};
  check(ground_hit(ground,.1,origin,ray,hit));
  check(std::abs(hit.x-1.25)<1e-9 && hit.z==-2.5);
  check(plane_hit(origin,ray,0.,flat) && std::abs(flat.x-hit.x-1.25)<1e-9);
  check(ground_hit(ground,.1,{1.25,0.,5.},{0.,0.,-1.},hit) && hit.x==1.25);
  check(!ground_hit({},.1,origin,ray,hit));
  check(!ground_hit({{1.,0.,-2.5}},.1,origin,ray,hit));
  check(!ground_hit(ground,.1,origin,{1.,0.,0.},hit));
  check(!ground_hit(ground,.1,origin,{0.,0.,1.},hit));
  check(ground_hit({{0.,0.,-2.},{0.,0.,-1.}},.1,{0.,0.,4.},{0.,0.,-1.},hit) && hit.z==-1.);
  check(!ground_hit(ground,0.,origin,ray,hit));
  std::cout << "10 ground-picking checks passed\n";
}
