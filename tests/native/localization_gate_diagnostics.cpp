#include "LocalizationGate.h"
#include <cassert>

int main() {
    LocalizationGate gate;
    gate.enabled = true;
    gate.confirmationMatches = 1;
    gate.regions = {{{-10,-10},{10,-10},{10,10},{-10,10}}};
    Eigen::Affine3f identity = Eigen::Affine3f::Identity();
    assert(gate.begin(identity));
    assert(gate.accept(identity,identity,identity,.9,.01));
    assert(gate.rejectionMask == 0);
    assert(gate.begin(identity));
    assert(!gate.accept(identity,identity,identity,.2,.3));
    assert(gate.reason == "coverage_match_quality_rejected");
    assert(gate.rejectionMask == (2u | 4u));
    Eigen::Affine3f moved=identity;
    moved.translation().x()=1.;
    assert(gate.begin(identity));
    assert(!gate.accept(identity,moved,identity,.9,.01));
    assert(gate.rejectionMask == 16u);
    Eigen::Affine3f tilted=identity;
    tilted.linear()=Eigen::AngleAxisf(5.3f*3.14159265358979323846f/180.f,Eigen::Vector3f::UnitX()).toRotationMatrix();
    assert(gate.begin(identity));
    assert(!gate.accept(identity,tilted,identity,.94,.04));
    assert(gate.rejectionMask == 128u);
    gate.maxGravity=6.*3.14159265358979323846/180.;
    assert(gate.begin(identity));
    assert(gate.accept(identity,tilted,identity,.94,.04));
}
