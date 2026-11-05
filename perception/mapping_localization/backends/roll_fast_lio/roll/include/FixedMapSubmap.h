#pragma once
#ifdef SOCIAL_NAV_NATIVE_ROLL
#include "roll_support.h"
#else
#include "utility.h"
#endif
#include <condition_variable>
#include <thread>

// One worker owns all construction buffers, filters and transformed-frame cache.
// Only immutable, fully built geometry crosses to the matching thread.
template<class Pose>
class FixedMapSubmap {
public:
    using Cloud = pcl::PointCloud<PointType>;
    using Tree = pcl::KdTreeFLANN<PointType>;
    struct Result {
        uint64_t epoch = 0, builds = 0, misses = 0;
        Eigen::Vector3f center;
        std::vector<int> ids;
        Cloud::Ptr corners{new Cloud}, surfaces{new Cloud};
        Cloud::Ptr cornerDS{new Cloud}, surfaceDS{new Cloud};
        Tree::Ptr cornerTree{new Tree}, surfaceTree{new Tree};
        double buildMs = 0;
    };
    using ResultPtr = std::shared_ptr<Result>;

    FixedMapSubmap(const pcl::PointCloud<Pose>& poses,
                   const std::vector<Cloud::Ptr>& corners, const std::vector<Cloud::Ptr>& surfaces,
                   float radius, float poseLeaf, float cornerLeaf, float surfaceLeaf,
                   size_t maxEntries, size_t maxBytes)
        : poses_(poses), corners_(corners), surfaces_(surfaces), radius_(radius),
          poseLeaf_(poseLeaf), cornerLeaf_(cornerLeaf), surfaceLeaf_(surfaceLeaf),
          maxEntries_(maxEntries), maxBytes_(maxBytes)
    {
        positions_.reset(new Cloud);
        for (const auto& pose : poses_) {
            PointType p; p.x = pose.x; p.y = pose.y; p.z = pose.z; p.intensity = pose.intensity;
            positions_->push_back(p);
        }
        poseTree_.setInputCloud(positions_);
        worker_ = std::thread(&FixedMapSubmap::run, this);
    }
    ~FixedMapSubmap() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        ready_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    FixedMapSubmap(const FixedMapSubmap&) = delete;
    FixedMapSubmap& operator=(const FixedMapSubmap&) = delete;

    void request(const Eigen::Vector3f& center, uint64_t epoch) {
        std::lock_guard<std::mutex> lock(mutex_);
        center_ = center; epoch_ = epoch; pending_ = true; result_.reset();
        ready_.notify_all();
    }
    ResultPtr take(uint64_t epoch, bool wait) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (wait) ready_.wait(lock, [&] { return stop_ || error_ || (result_ && result_->epoch == epoch); });
        if (error_) std::rethrow_exception(error_);
        if (!result_ || result_->epoch != epoch) return {};
        auto result = result_; result_.reset(); return result;
    }

private:
    pcl::PointCloud<Pose> poses_;
    std::vector<Cloud::Ptr> corners_, surfaces_;
    Cloud::Ptr positions_;
    Tree poseTree_;
    float radius_, poseLeaf_, cornerLeaf_, surfaceLeaf_;
    size_t maxEntries_, maxBytes_, bytes_ = 0;
    uint64_t sequence_ = 0, builds_ = 0, misses_ = 0;
    struct Cached { Cloud::Ptr corners, surfaces; uint64_t access; size_t bytes; };
    std::map<int, Cached> cache_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::thread worker_;
    bool stop_ = false, pending_ = false;
    uint64_t epoch_ = 0;
    Eigen::Vector3f center_;
    ResultPtr result_;
    std::exception_ptr error_;

    Cloud::Ptr transform(const Cloud::Ptr& input, int id) {
        const auto& p = poses_[id];
        const auto t = pcl::getTransformation(p.x, p.y, p.z, p.roll, p.pitch, p.yaw);
        Cloud::Ptr out(new Cloud); out->resize(input->size());
        for (size_t i = 0; i < input->size(); ++i) {
            const auto& v = (*input)[i]; auto& q = (*out)[i];
            q.x = t(0,0)*v.x + t(0,1)*v.y + t(0,2)*v.z + t(0,3);
            q.y = t(1,0)*v.x + t(1,1)*v.y + t(1,2)*v.z + t(1,3);
            q.z = t(2,0)*v.x + t(2,1)*v.y + t(2,2)*v.z + t(2,3);
            q.intensity = v.intensity;
        }
        return out;
    }
    ResultPtr build(const Eigen::Vector3f& center, uint64_t epoch) {
        TicToc timer;
        ResultPtr out(new Result); out->center = center; out->epoch = epoch;
        PointType p; p.x = center.x(); p.y = center.y(); p.z = center.z();
        std::vector<int> ids; std::vector<float> distances;
        poseTree_.radiusSearch(p, radius_, ids, distances);
        Cloud::Ptr nearby(new Cloud), sampled(new Cloud);
        for (int id : ids) nearby->push_back((*positions_)[id]);
        pcl::VoxelGrid<PointType> filter;
        if (!nearby->empty()) {
            filter.setLeafSize(poseLeaf_, poseLeaf_, poseLeaf_);
            filter.setInputCloud(nearby); filter.filter(*sampled);
        }
        for (const auto& pose : *sampled) {
            poseTree_.nearestKSearch(pose, 1, ids, distances);
            out->ids.push_back(static_cast<int>((*positions_)[ids[0]].intensity));
        }
        std::sort(out->ids.begin(), out->ids.end()); // Preserve duplicate weighting.
        for (int id : out->ids) {
            auto found = cache_.find(id);
            if (found == cache_.end()) {
                ++misses_;
                auto corner = transform(corners_[id], id), surface = transform(surfaces_[id], id);
                size_t bytes = (corner->size()+surface->size())*sizeof(PointType);
                while (!cache_.empty() && (cache_.size() >= maxEntries_ || bytes_+bytes > maxBytes_)) {
                    auto oldest = std::min_element(cache_.begin(), cache_.end(), [](const auto& a, const auto& b) {
                        return a.second.access < b.second.access;
                    });
                    bytes_ -= oldest->second.bytes; cache_.erase(oldest);
                }
                if (bytes > maxBytes_) {
                    *out->corners += *corner; *out->surfaces += *surface; continue;
                }
                found = cache_.emplace(id, Cached{corner, surface, ++sequence_, bytes}).first;
                bytes_ += bytes;
            }
            found->second.access = ++sequence_;
            *out->corners += *found->second.corners; *out->surfaces += *found->second.surfaces;
        }
        filter.setLeafSize(cornerLeaf_, cornerLeaf_, cornerLeaf_);
        if (!out->corners->empty()) {
            filter.setInputCloud(out->corners); filter.filter(*out->cornerDS);
            if (!out->cornerDS->empty()) out->cornerTree->setInputCloud(out->cornerDS);
        }
        filter.setLeafSize(surfaceLeaf_, surfaceLeaf_, surfaceLeaf_);
        if (!out->surfaces->empty()) {
            filter.setInputCloud(out->surfaces); filter.filter(*out->surfaceDS);
            if (!out->surfaceDS->empty()) out->surfaceTree->setInputCloud(out->surfaceDS);
        }
        out->builds = ++builds_; out->misses = misses_; out->buildMs = timer.toc();
        return out;
    }
    void run() {
        while (true) {
            Eigen::Vector3f center; uint64_t epoch;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return stop_ || pending_; });
                if (stop_) return;
                center = center_; epoch = epoch_; pending_ = false;
            }
            try {
                auto result = build(center, epoch);
                std::lock_guard<std::mutex> lock(mutex_);
                if (!pending_ && epoch_ == epoch) result_ = result;
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_); error_ = std::current_exception();
            }
            ready_.notify_all();
        }
    }
};
