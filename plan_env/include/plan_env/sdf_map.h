#ifndef _SDF_MAP_H
#define _SDF_MAP_H

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include <queue>
#include <ros/ros.h>
#include <tuple>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include<omp.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include "plan_env/random_mapping_method.h"
#include "plan_env/raycast.h"  
#include "plan_env/map_ros.h"   
using namespace std;

namespace cv {
class Mat;
}

class RayCaster;

class random_mapping_method;

namespace fast_planner {
struct MapParam;
struct MapData;
class MapROS;

class SDFMap {
public:
  // SDFMap();
  SDFMap() : rmm(200, "sin", 2.5, "cla") {}
  ~SDFMap();

  random_mapping_method rmm;
  // void setMapROS(MapROS* mr);
  enum OCCUPANCY { UNKNOWN, FREE, OCCUPIED };

  void initMap(ros::NodeHandle& nh);
  void inputPointCloud(const pcl::PointCloud<pcl::PointXYZ>& points, const int& point_num,
                       const Eigen::Vector3d& camera_pos);

  pcl::PointCloud<pcl::PointXYZ> map_surface_value;

  Eigen::MatrixXd point_local_map1;
  Eigen::MatrixXd point_local_map1_black;
  Eigen::VectorXd Y_test_predicted;
  Eigen::VectorXd Y_test_predicted_combined;
  Eigen::MatrixXd point_local_map1_combined;
  Eigen::VectorXd Y_test_combined;
  Eigen::VectorXd Y_test_predicted_black; // Used for predicting occupancy states in blind spots
  Eigen::MatrixXd point_p1; // Used to store point coordinates
  Eigen::MatrixXd point_p1_black; // Used to store blind spot coordinates

  // Point cloud voxelization and downsampling
  void processGridPoints(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                          std::unordered_map<int, int>& grid_count,
                          std::set<int>& unique_adrs);

  void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
  void indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos);
  void boundIndex(Eigen::Vector3i& id);
  int toAddress(const Eigen::Vector3i& id);
  int toAddress(const int& x, const int& y, const int& z);
  void addressToIndex_occu_change_rate_training_buffer(const int& adr, int& x, int& y, int& z);
  bool isInMap(const Eigen::Vector3d& pos);
  bool isInMap(const Eigen::Vector3i& idx);
  bool isInBox(const Eigen::Vector3i& id);
  bool isInBox(const Eigen::Vector3d& pos);
  void boundBox(Eigen::Vector3d& low, Eigen::Vector3d& up);
  int getOccupancy(const Eigen::Vector3d& pos);
  int getOccupancy(const Eigen::Vector3i& id);
  void setOccupied(const Eigen::Vector3d& pos, const int& occ = 1);
  int getInflateOccupancy(const Eigen::Vector3d& pos);
  int getInflateOccupancy(const Eigen::Vector3i& id);
  double getDistance(const Eigen::Vector3d& pos);
  double getDistance(const Eigen::Vector3i& id);
  // double getDistWithGrad_esdf(const Eigen::Vector3d& pos, Eigen::Vector3d& grad);
  double getDistWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad);
  void updateESDF3d();
  void resetBuffer();
  void resetBuffer(const Eigen::Vector3d& min, const Eigen::Vector3d& max);

  void getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size);
  void getBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax);
  void getUpdatedBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax, bool reset = false);
  double getResolution();
  int getVoxelNum();

private:
    // void publishPredictions(const Eigen::VectorXd& predictions,
    //                        const Eigen::MatrixXd& positions);
    // Define the training data structure for occupancy change rate
    struct SimpleTrainingData {
        Eigen::Vector3d pos;
        int occupancy;
    };
    // Use a fixed-size vector to store training data
    std::vector<SimpleTrainingData> training_buffer;
    
    // GTX 1650 optimization: skip frame training to ensure smoothness (fix trajectory stutter)
    int training_frame_counter_ = 0;
    const int TRAIN_EVERY_N_FRAMES = 3;  // Train every 3 frames to avoid blocking
    
  void updateOccupancyChangeRateInfo();

  void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud, 
                         ros::Publisher& publisher);
  void publishGridPoint(const Eigen::Vector3d& pos);
  void publishPredictions(const Eigen::VectorXd& predictions,
                          const Eigen::MatrixXd& positions);
  ros::Publisher training_points_pub;     
  void publishPredictions_new(const Eigen::VectorXd& predictions,
                              const Eigen::MatrixXd& positions);
  void trainAndPredictOccupancy(
      std::vector<SimpleTrainingData>& training_buffer,
      const pcl::PointCloud<pcl::PointXYZ>& map_surface_value,
      random_mapping_method& rmm);

  const size_t MAX_TRAINING_SIZE = 256; // Increase from 100 to 256, compatible with max_rows=256

  void clearAndInflateLocalMap();
  void inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts);
  void setCacheOccupancy(const int& adr, const int& occ);
  void setCacheOccupancy1(const int& adr, const int& occ);
  Eigen::Vector3d closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt);
  template <typename F_get_val, typename F_set_val>
  void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim);

  unique_ptr<MapParam> mp_;
  unique_ptr<MapData> md_;
  unique_ptr<MapROS> mr_;
  unique_ptr<RayCaster> caster_;

  friend MapROS;
  // MapROS* mapros;

public:
  typedef std::shared_ptr<SDFMap> Ptr;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MapParam {
  // map properties
  Eigen::Vector3d map_origin_, map_size_;
  Eigen::Vector3d map_min_boundary_, map_max_boundary_;
  Eigen::Vector3i map_voxel_num_;
  double resolution_, resolution_inv_;
  double obstacles_inflation_;
  double virtual_ceil_height_, ground_height_, blue_region_z_;
  Eigen::Vector3i box_min_, box_max_;
  Eigen::Vector3d box_mind_, box_maxd_;
  double default_dist_;
  bool optimistic_, signed_dist_;
  // map fusion
  double p_hit_, p_miss_, p_min_, p_max_, p_occ_;  // occupancy probability
  double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_, min_occupancy_log_;  // logit
  double max_ray_length_;
  double local_bound_inflate_;
  int local_map_margin_;
  double unknown_flag_;
};

struct MapData {
  std::vector<double> occupancy_buffer_;            
  std::vector<char> occupancy_buffer_inflate_;       
  std::vector<double> distance_buffer_neg_;          
  std::vector<double> distance_buffer_;            
  std::vector<double> tmp_buffer1_;               
  std::vector<double> tmp_buffer2_;                   

  std::vector<short> count_hit_;                    
  std::vector<short> count_miss_;                    
  std::vector<short> count_hit_and_miss_;             
  std::vector<char> flag_rayend_;                     
  std::vector<char> flag_visited_;                  
  char raycast_num_;                               
  std::queue<int> cache_voxel_;                      
  
  Eigen::Vector3i local_bound_min_;                  
  Eigen::Vector3i local_bound_max_;                
  Eigen::Vector3d update_min_;                       
  Eigen::Vector3d update_max_;                      
  bool reset_updated_box_;                            

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW                     
};

//Grid point cloud, point cloud downsampling
inline void SDFMap::processGridPoints(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                            std::unordered_map<int, int>& grid_count,
                            std::set<int>& unique_adrs) {
    for (const auto& pt : cloud.points) {
        Eigen::Vector3d pt_w(pt.x, pt.y, pt.z);
        Eigen::Vector3i idx;
        posToIndex(pt_w, idx);
            
        int adr = idx[0] + idx[1] * mp_->map_voxel_num_[0] + 
                idx[2] * mp_->map_voxel_num_[0] * mp_->map_voxel_num_[1];
            
        grid_count[adr]++;
        unique_adrs.insert(adr);
    }
}

inline void SDFMap::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i)
    id(i) = floor((pos(i) - mp_->map_origin_(i)) * mp_->resolution_inv_);
}

inline void SDFMap::indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i)
    pos(i) = (id(i) + 0.5) * mp_->resolution_ + mp_->map_origin_(i);
}

inline void SDFMap::boundIndex(Eigen::Vector3i& id) {
  Eigen::Vector3i id1;
  id1(0) = max(min(id(0), mp_->map_voxel_num_(0) - 1), 0);
  id1(1) = max(min(id(1), mp_->map_voxel_num_(1) - 1), 0);
  id1(2) = max(min(id(2), mp_->map_voxel_num_(2) - 1), 0);
  id = id1;
}

inline int SDFMap::toAddress(const int& x, const int& y, const int& z) {
  return x * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2) + y * mp_->map_voxel_num_(2) + z;
}

inline int SDFMap::toAddress(const Eigen::Vector3i& id) {
  return toAddress(id[0], id[1], id[2]);
}

// Occupancy change rate training region part address to index implementation part
inline void SDFMap::addressToIndex_occu_change_rate_training_buffer(const int& adr, int& x, int& y, int& z) {
  // Get the size of each layer and each row
  const int layer_size = mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2);
  const int row_size = mp_->map_voxel_num_(2);
  // Calculate x (layer)
  x = adr / layer_size;
  // Calculate y (row)
  const int remainder_y = adr % layer_size;
  y = remainder_y / row_size;
  // Calculate z (column)
  z = remainder_y % row_size;
}

inline bool SDFMap::isInMap(const Eigen::Vector3d& pos) {
  double boundary_buffer = 0.1;  // Add a small buffer to exclude floating point errors

  // Use the defined maximum and minimum values of x and y as the boundary of the map
  if (pos(0) < -7.97 + boundary_buffer || pos(0) > 7.67 - boundary_buffer || 
      pos(1) < -15.12 + boundary_buffer || pos(1) > 15.18 - boundary_buffer ||
      pos(2) < mp_->map_min_boundary_(2) + boundary_buffer || pos(2) > mp_->map_max_boundary_(2) - boundary_buffer) {
    return false;
  }
  return true;
}

inline bool SDFMap::isInMap(const Eigen::Vector3i& idx) {
  if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) return false;
  if (idx(0) > mp_->map_voxel_num_(0) - 1 || idx(1) > mp_->map_voxel_num_(1) - 1 ||
      idx(2) > mp_->map_voxel_num_(2) - 1)
    return false;
  return true;
}

inline bool SDFMap::isInBox(const Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i) {
    if (id[i] < mp_->box_min_[i] || id[i] >= mp_->box_max_[i]) {
      return false;
    }
  }
  return true;
}

inline bool SDFMap::isInBox(const Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i) {
    if (pos[i] <= mp_->box_mind_[i] || pos[i] >= mp_->box_maxd_[i]) {
      return false;
    }
  }
  return true;
}

inline void SDFMap::boundBox(Eigen::Vector3d& low, Eigen::Vector3d& up) {
  for (int i = 0; i < 3; ++i) {
    low[i] = max(low[i], mp_->box_mind_[i]);
    up[i] = min(up[i], mp_->box_maxd_[i]);
  }
}

inline int SDFMap::getOccupancy(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  double occ = md_->occupancy_buffer_[toAddress(id)];
  if (occ < mp_->clamp_min_log_ - 1e-3) return UNKNOWN;
  if (occ > mp_->min_occupancy_log_) return OCCUPIED;
  return FREE;
}

inline int SDFMap::getOccupancy(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getOccupancy(id);
}

inline void SDFMap::setOccupied(const Eigen::Vector3d& pos, const int& occ) {
  if (!isInMap(pos)) return;
  Eigen::Vector3i id;
  posToIndex(pos, id);
  md_->occupancy_buffer_inflate_[toAddress(id)] = occ;
}

inline int SDFMap::getInflateOccupancy(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  return int(md_->occupancy_buffer_inflate_[toAddress(id)]);
}

inline int SDFMap::getInflateOccupancy(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getInflateOccupancy(id);
}

inline double SDFMap::getDistance(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  return md_->distance_buffer_[toAddress(id)];
}

inline double SDFMap::getDistance(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getDistance(id);
}

inline void SDFMap::inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts) {
  int num = 0;

  /* ---------- + shape inflate ---------- */
  // for (int x = -step; x <= step; ++x)
  // {
  //   if (x == 0)
  //     continue;
  //   pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1), pt(2));
  // }
  // for (int y = -step; y <= step; ++y)
  // {
  //   if (y == 0)
  //     continue;
  //   pts[num++] = Eigen::Vector3i(pt(0), pt(1) + y, pt(2));
  // }
  // for (int z = -1; z <= 1; ++z)
  // {
  //   pts[num++] = Eigen::Vector3i(pt(0), pt(1), pt(2) + z);
  // }

  /* ---------- all inflate ---------- */
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -step; z <= step; ++z) {
        pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z);
      }
}
}
#endif
