#pragma once
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// Coverage is a localization trust region, never a free-space assertion.
class LocalizationGate {
public:
    bool enabled = false;
    std::vector<std::vector<Eigen::Vector2f>> regions;
    double reentryMargin = .5, minInlier = .55, maxResidual = .10;
    double maxStep = .75, maxAngle = .15, maxGravity = .0872665;
    double consistencyDistance = .25, consistencyAngle = .035;
    int confirmationMatches = 5;
    std::string reason;

    void reset() { tracking = false; outside = false; pending = 0; anchorReady = false; }
    float clearance(const Eigen::Vector3f& position) const {
        float best = -INFINITY;
        const Eigen::Vector2f p = position.head<2>();
        for (const auto& polygon : regions) {
            bool inside = false;
            float distance = INFINITY;
            for (size_t i = 0; i < polygon.size(); ++i) {
                const auto& a = polygon[i]; const auto& b = polygon[(i+1)%polygon.size()];
                Eigen::Vector2f d = b-a;
                float t = d.squaredNorm() > 0 ? std::max(0.f, std::min(1.f, (p-a).dot(d)/d.squaredNorm())) : 0;
                distance = std::min(distance, (p-a-t*d).norm());
                if ((a.y()>p.y()) != (b.y()>p.y()))
                    inside ^= p.x() < (b.x()-a.x())*(p.y()-a.y())/(b.y()-a.y())+a.x();
            }
            best = std::max(best, inside ? distance : -distance);
        }
        return best;
    }
    bool begin(const Eigen::Affine3f& prediction) {
        reason.clear();
        if (!enabled) return true;
        const float distance = clearance(prediction.translation());
        if (distance < -1e-5f) { outside = true; tracking = false; pending = 0; }
        if (outside && distance < reentryMargin) {
            reason = "outside_prior_coverage"; return false;
        }
        return true;
    }
    bool accept(const Eigen::Affine3f& prediction, const Eigen::Affine3f& candidate,
                const Eigen::Affine3f& local, double inlier, double residual) {
        if (!enabled) return std::isfinite(residual) && residual >= 0 && std::isfinite(inlier) && inlier >= .3;
        const Eigen::Affine3f correction = candidate * local.inverse();
        if (!anchorReady) { anchor = prediction * local.inverse(); anchorReady = true; }
        const float angle = Eigen::AngleAxisf(prediction.linear().transpose()*candidate.linear()).angle();
        const float gravity = std::acos(std::max(-1.f, std::min(1.f,
            (anchor.linear()*Eigen::Vector3f::UnitZ()).dot(correction.linear()*Eigen::Vector3f::UnitZ()))));
        if (!candidate.matrix().allFinite() || !std::isfinite(inlier) || !std::isfinite(residual) || residual < 0 ||
                residual > maxResidual || inlier < minInlier || clearance(candidate.translation()) < -1e-5f ||
                (candidate.translation()-prediction.translation()).norm() > maxStep ||
                angle > maxAngle || gravity > maxGravity) {
            tracking = false; pending = 0; reason = "coverage_match_quality_rejected"; return false;
        }
        if (!tracking) {
            if (pending && ((correction.translation()-previous.translation()).norm() > consistencyDistance ||
                    Eigen::AngleAxisf(previous.linear().transpose()*correction.linear()).angle() > consistencyAngle))
                pending = 0;
            previous = correction;
            if (++pending < confirmationMatches) { reason = "coverage_match_confirming"; return false; }
        }
        tracking = true; outside = false; pending = 0; return true;
    }
    void missing() { if (enabled) { tracking = false; pending = 0; reason = "coverage_match_unavailable"; } }
private:
    bool tracking = false, outside = false, anchorReady = false;
    int pending = 0;
    Eigen::Affine3f previous = Eigen::Affine3f::Identity(), anchor = Eigen::Affine3f::Identity();
};
