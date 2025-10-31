/*
 * @brief Implementation of the SDF Map integrated with the Random Mapping Method (RMM).
 *
 * This file extends a standard Signed Distance Field (SDF) map implementation.
 * The primary addition is the integration of the Random Mapping Method (RMM)
 * for managing and inferring the occupancy grid.
 *
 * Key additional functionalities include:
 * 1. Online training of the RMM model using live sensor data (point clouds).
 * 2. Real-time prediction of occupancy states to perform map completion and
 *    inference in unobserved or occluded areas.
 * 3. Invocation of CUDA-accelerated functions for high-performance computation
 *    of the RMM's feature mapping, training, and prediction steps.
 *
 * This hybrid approach combines the utility of an SDF for gradient-based
 * planning with an intelligent, learning-based model for creating complete
 * and coherent environment representations.
 */
#include "plan_env/sdf_map.h"
#include "plan_env/map_ros.h"
#include "plan_env/raycast.h"
#include "plan_env/random_mapping_method.h"
#include <iostream>
#include <fstream>
#include <algorithm> // for std::min and std::max
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <omp.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>

int bufsize;

namespace fast_planner {

// Constructor: Initializes the SDF map object
// SDFMap::SDFMap() {
//   // mapros = new MapROS(); // Dynamically allocate a MapROS instance
// }

// Destructor: Releases resources held by the SDF map object
SDFMap::~SDFMap() {
  // delete mapros;
}

// void SDFMap::setMapROS(MapROS* mr) {
//   this->mapros = mr;
// }

// Initialize SDF map
void SDFMap::initMap(ros::NodeHandle& nh) {
  mp_.reset(new MapParam); // Map parameters
  md_.reset(new MapData); // Map data
  mr_.reset(new MapROS);  // ROS interface related

  // Params of map properties// Map properties parameters
  double x_size, y_size, z_size;
  nh.param("sdf_map/resolution", mp_->resolution_, -1.0);
  nh.param("sdf_map/map_size_x", x_size, -1.0);
  nh.param("sdf_map/map_size_y", y_size, -1.0);
  nh.param("sdf_map/map_size_z", z_size, -1.0);
  nh.param("sdf_map/obstacles_inflation", mp_->obstacles_inflation_, -1.0);
  nh.param("sdf_map/local_bound_inflate", mp_->local_bound_inflate_, 1.0);
  nh.param("sdf_map/local_map_margin", mp_->local_map_margin_, 1);
  nh.param("sdf_map/ground_height", mp_->ground_height_, 1.0);
  nh.param("sdf_map/default_dist", mp_->default_dist_, 5.0);
  nh.param("sdf_map/optimistic", mp_->optimistic_, true);
  nh.param("sdf_map/signed_dist", mp_->signed_dist_, false);
  nh.param("sdf_map/blue_region_z", mp_->blue_region_z_, -0.1);

 // Determine map resolution and size
  mp_->local_bound_inflate_ = max(mp_->resolution_, mp_->local_bound_inflate_);
  mp_->resolution_inv_ = 1 / mp_->resolution_;
  mp_->map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_->ground_height_);
  mp_->map_size_ = Eigen::Vector3d(x_size, y_size, z_size);
  for (int i = 0; i < 3; ++i)
    mp_->map_voxel_num_(i) = ceil(mp_->map_size_(i) / mp_->resolution_);
  mp_->map_min_boundary_ = mp_->map_origin_;
  std::cout << "mp_-> map_min_boundary_:" << mp_-> map_min_boundary_ << std::endl;
  std::cout << "mp_-> map_size_:" << mp_-> map_size_ << std::endl;
  mp_->map_max_boundary_ = mp_->map_origin_ + mp_->map_size_;
  std::cout << "mp_-> map_max_boundary_:" << mp_-> map_max_boundary_ << std::endl;

  // Params of raycasting-based fusion
  nh.param("sdf_map/p_hit", mp_->p_hit_, 0.70);
  nh.param("sdf_map/p_miss", mp_->p_miss_, 0.35);
  nh.param("sdf_map/p_min", mp_->p_min_, 0.12);
  nh.param("sdf_map/p_max", mp_->p_max_, 0.97);
  nh.param("sdf_map/p_occ", mp_->p_occ_, 0.80);
  nh.param("sdf_map/max_ray_length", mp_->max_ray_length_, -0.1);
  nh.param("sdf_map/virtual_ceil_height", mp_->virtual_ceil_height_, -0.1);

// Logit transformation of probability, used for subsequent probability update
  auto logit = [](const double& x) { return log(x / (1 - x)); };
  mp_->prob_hit_log_ = logit(mp_->p_hit_);
  mp_->prob_miss_log_ = logit(mp_->p_miss_);
  mp_->clamp_min_log_ = logit(mp_->p_min_);
  mp_->clamp_max_log_ = logit(mp_->p_max_);
  mp_->min_occupancy_log_ = logit(mp_->p_occ_);
  mp_->unknown_flag_ = 0.01;

  // Print logit transformation results
  cout << "hit: " << mp_->prob_hit_log_ << ", miss: " << mp_->prob_miss_log_
       << ", min: " << mp_->clamp_min_log_ << ", max: " << mp_->clamp_max_log_
       << ", thresh: " << mp_->min_occupancy_log_ << endl;

  // Initialize data buffer of map
  int buffer_size = mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2);
  md_->occupancy_buffer_ = vector<double>(buffer_size, mp_->clamp_min_log_ - mp_->unknown_flag_);
  md_->occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);
  md_->distance_buffer_neg_ = vector<double>(buffer_size, mp_->default_dist_);
  md_->distance_buffer_ = vector<double>(buffer_size, mp_->default_dist_);
  md_->count_hit_and_miss_ = vector<short>(buffer_size, 0);
  md_->count_hit_ = vector<short>(buffer_size, 0);
  md_->count_miss_ = vector<short>(buffer_size, 0);
  md_->flag_rayend_ = vector<char>(buffer_size, -1);
  md_->flag_visited_ = vector<char>(buffer_size, -1);
  md_->tmp_buffer1_ = vector<double>(buffer_size, 0);
  md_->tmp_buffer2_ = vector<double>(buffer_size, 0);
  md_->raycast_num_ = 0;
  md_->reset_updated_box_ = true;
  md_->update_min_ = md_->update_max_ = Eigen::Vector3d(0, 0, 0);
  bufsize=buffer_size;

  // Try retriving bounding box of map, set box to map size if not specified
  vector<string> axis = { "x", "y", "z" };
  for (int i = 0; i < 3; ++i) {
    nh.param("sdf_map/box_min_" + axis[i], mp_->box_mind_[i], mp_->map_min_boundary_[i]);
    nh.param("sdf_map/box_max_" + axis[i], mp_->box_maxd_[i], mp_->map_max_boundary_[i]);
  }
  posToIndex(mp_->box_mind_, mp_->box_min_);
  posToIndex(mp_->box_maxd_, mp_->box_max_);

  // Initialize ROS wrapper
  mr_->setMap(this);
  mr_->node_ = nh;
  mr_->init();

  caster_.reset(new RayCaster);
  caster_->setParams(mp_->resolution_, mp_->map_origin_);
}

// Reset map buffers
void SDFMap::resetBuffer() {
  resetBuffer(mp_->map_min_boundary_, mp_->map_max_boundary_);
  md_->local_bound_min_ = Eigen::Vector3i::Zero();
  md_->local_bound_max_ = mp_->map_voxel_num_ - Eigen::Vector3i::Ones();
}

// Reset the map buffer within a specified region
void SDFMap::resetBuffer(const Eigen::Vector3d& min_pos, const Eigen::Vector3d& max_pos) {
  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);
  boundIndex(min_id);
  boundIndex(max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        md_->occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
        md_->distance_buffer_[toAddress(x, y, z)] = mp_->default_dist_;
      }
}

/*----------------CUDA acceleration part of feature_mapping------------------*/
// Declare CUDA function
extern "C" void feature_mapping_cuda(
    double* h_dataSet, int m, int initial_dim, int n,
    double* h_randomWeights, double* h_randomBias,
    double* h_randomSet,
    int actiFuncCode, double scaleRate);

// In the random_mapping_method class, modify the feature_mapping function
Eigen::MatrixXd random_mapping_method::feature_mapping(Eigen::MatrixXd& dataSet) {
    int initial_dim = dataSet.cols();
    int m = dataSet.rows();
    int n = targetDimen;

    // GTX 1650 optimization: only generate random weights the first time, then reuse 
    static bool weights_initialized = false;
    if (!weights_initialized || randomWeights_occu.rows() != initial_dim || randomWeights_occu.cols() != n) {
        randomWeights_occu = Eigen::MatrixXd::Random(initial_dim, n) * scaleRate;
        randomBias_occu = Eigen::MatrixXd::Random(1, n) * scaleRate;
        weights_initialized = true;
    }

    // Allocate result matrix
    Eigen::MatrixXd randomSet(m, n);

    int actiFuncCode;
    if (actiFunc == "sigmoid") {
        actiFuncCode = 0;
    } else if (actiFunc == "sin") {
        actiFuncCode = 1;
    } else if (actiFunc == "linear") {
        actiFuncCode = 2;
    } else if (actiFunc == "tanh") {
        actiFuncCode = 3;
    } else {
        actiFuncCode = 2; 
    }

    // Call CUDA function
    fast_planner::feature_mapping_cuda(
        const_cast<double*>(dataSet.data()), m, initial_dim, n,
        // randomWeights.data(), randomBias.data(),
        randomWeights_occu.data(), randomBias_occu.data(),
        randomSet.data(),
        actiFuncCode, scaleRate
    );

    return randomSet;
}

// Feature mapping function with trained parameters:
Eigen::MatrixXd random_mapping_method::feature_mapping_with_params(
    Eigen::MatrixXd& dataSet,
    const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& bias) {
    
    int initial_dim = dataSet.cols();
    int m = dataSet.rows();
    int n = targetDimen;

    Eigen::MatrixXd randomSet(m, n);

    int actiFuncCode;
    if (actiFunc == "sigmoid") {
        actiFuncCode = 0;
    } else if (actiFunc == "sin") {
        actiFuncCode = 1;
    } else if (actiFunc == "linear") {
        actiFuncCode = 2;
    } else if (actiFunc == "tanh") {
        actiFuncCode = 3;
    } else {
        actiFuncCode = 2;
    }

    fast_planner::feature_mapping_cuda(
        const_cast<double*>(dataSet.data()), m, initial_dim, n,
        const_cast<double*>(weights.data()),
        const_cast<double*>(bias.data()),
        randomSet.data(),
        actiFuncCode, scaleRate
    );

    return randomSet;
}

/*-----------------------------CUDA acceleration train function------------------------------*/
extern "C" void train_occupancy_change_rate_cuda(
    double* h_X,
    double* h_Y,
    double* h_theta,
    double* h_randomWeights,
    double* h_randomBias,
    int rows,
    int cols,
    int targetDimen,
    int num_epochs,
    double learning_rate,
    const char* activation_func
);

// Train occupancy change rate CUDA implementation using gradient descent
void random_mapping_method::train_occupancy_change_rate(
    Eigen::MatrixXd X, 
    Eigen::VectorXd Y, 
    int num_epochs
) {
    try {
        // Check input data
        if (X.rows() == 0 || Y.rows() == 0) {
            std::cerr << "Error: Empty input data!" << std::endl;
            return;
        }
        
        if (X.rows() != Y.rows()) {
            std::cerr << "Error: Input data X and Y have mismatched dimensions!" << std::endl;
            return;
        }

        // Check if input data contains NaN or Inf
        if (!X.allFinite()) {
            std::cerr << "Error: Input X contains NaN or Inf values!" << std::endl;
            return;
        }
        
        if (!Y.allFinite()) {
            std::cerr << "Error: Input Y contains NaN or Inf values!" << std::endl;
            return;
        }

        // Save original training data
        X_training_data_occu_change_rate = X;
        Y_training_data_occu_change_rate = Y;
        training_samples_occu_change_rate = X.rows();
        feature_dimension_occu_change_rate = X.cols();

        // Use minimum-maximum normalization
        Eigen::VectorXd X_min_occu = X.colwise().minCoeff();
        // std::cout << "X_min_occu size: " << X_min_occu.size() << std::endl;//200
        Eigen::VectorXd X_max_occu = X.colwise().maxCoeff();
        Eigen::VectorXd X_range_occu = X_max_occu - X_min_occu;

        // Avoid division by zero
        for (int i = 0; i < X_range_occu.size(); i++) {
            if (X_range_occu(i) < 1e-10) {
                X_range_occu(i) = 1.0;
            }
        }
        
        // Normalize to [-1, 1] range
        X_normalized_occu_change_rate = 2.0 * ((X.rowwise() - X_min_occu.transpose()).array().rowwise() / X_range_occu.transpose().array()) - 1.0;
        
        // Verify if X_normalized contains abnormal values
        if (!X_normalized_occu_change_rate.allFinite()) {
            std::cerr << "Error: X_normalized contains NaN or Inf after normalization!" << std::endl;
            return;
        }
        
        // Normalize Y
        double Y_min_occu = Y.minCoeff();
        Y_max_occu_change_rate = Y.maxCoeff();
        double Y_range_occu = Y_max_occu_change_rate - Y_min_occu;
        if (Y_range_occu < 1e-10) Y_range_occu = 1.0;
        
        Y_normalized_occu_change_rate = 2.0 * ((Y.array() - Y_min_occu) / Y_range_occu) - 1.0;
        
        // Verify if Y_normalized contains abnormal values
        if (!Y_normalized_occu_change_rate.allFinite()) {
            std::cerr << "Error: Y_normalized contains NaN or Inf after normalization!" << std::endl;
            return;
        }
        
        // Store normalization parameters
        X_min_occu_change_rate = X_min_occu;
        X_range_occu_change_rate = X_range_occu;
        Y_min_occu_change_rate = Y_min_occu;
        Y_range_occu_change_rate = Y_range_occu;

        // Initialize random weights and bias
        static bool weights_initialized = false;
        if (!weights_initialized)
        {
        randomWeights_occu = Eigen::MatrixXd::Random(feature_dimension_occu_change_rate, targetDimen) * scaleRate;
        randomBias_occu = Eigen::MatrixXd::Random(1, targetDimen) * scaleRate;
        weights_initialized = true;
        }
        
        // Initialize theta parameters
        // theta = Eigen::VectorXd::Random(targetDimen + 1) * 0.01;  // +1 for bias

        // Prepare training data
        Eigen::MatrixXd X_bias(training_samples_occu_change_rate, feature_dimension_occu_change_rate + 1);
        X_bias << X_normalized_occu_change_rate, Eigen::MatrixXd::Ones(training_samples_occu_change_rate, 1);

        // Call CUDA training function
        fast_planner::train_occupancy_change_rate_cuda(
            X_bias.data(),
            Y_normalized_occu_change_rate.data(),
            theta.data(),
            randomWeights_occu.data(),
            randomBias_occu.data(),
            training_samples_occu_change_rate,
            feature_dimension_occu_change_rate + 1,
            targetDimen,
            num_epochs,
            learning_rate_occu_change_rate,
            actiFunc.c_str()
        );
        // std::cout << "after theta_occu: " << theta << std::endl;

        // Calculate training score
        // double training_score = score_occupancy_change_rate(X, Y);
        // std::cout << "Training accuracy for occupancy change rate: " << training_score << std::endl;

        // Verify training results: check if theta contains NaN or Inf
        if (!theta.allFinite()) {
            std::cerr << "Error: Invalid values in trained parameters!" << std::endl;
            std::cerr << "Resetting theta to zero. This may indicate:" << std::endl;
            std::cerr << "  - Numerical instability during training" << std::endl;
            std::cerr << "  - Learning rate too high" << std::endl;
            std::cerr << "  - Input data issues" << std::endl;
            theta = Eigen::VectorXd::Zero(targetDimen + 1);
        }
        
        // Additional verification: check if random weights and bias contain invalid values
        if (!randomWeights_occu.allFinite()) {
            std::cerr << "Warning: Random weights contain invalid values, reinitializing..." << std::endl;
            randomWeights_occu = Eigen::MatrixXd::Random(feature_dimension_occu_change_rate, targetDimen) * scaleRate;
        }
        
        if (!randomBias_occu.allFinite()) {
            std::cerr << "Warning: Random bias contain invalid values, reinitializing..." << std::endl;
            randomBias_occu = Eigen::MatrixXd::Random(1, targetDimen) * scaleRate;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception in train_occupancy_change_rate: " << e.what() << std::endl;
        theta = Eigen::VectorXd::Zero(feature_dimension_occu_change_rate + 1);
    } catch (...) {
        std::cerr << "Unknown exception in train_occupancy_change_rate" << std::endl;
        theta = Eigen::VectorXd::Zero(feature_dimension_occu_change_rate + 1);
    }
}

/*--------------------------CUDA acceleration predict function--------------------------------*/
extern "C" void predict_occupancy_change_rate_cuda(
    const double* X_normalized,
    const double* theta,
    const double* weights,
    const double* bias,
    double* Y_predicted,
    int rows,
    int feature_cols,
    int target_dim,
    const char* activation_func
);

Eigen::VectorXd random_mapping_method::predict_occupancy_change_rate(const Eigen::MatrixXd& X2predict) {
    // Handle empty input
    if (X2predict.rows() == 0) {
        return Eigen::VectorXd();
    }

    // Verify input dimension
    // std::cout << "feature_dimension_occu_change_rate: " << feature_dimension_occu_change_rate << std::endl;
    if (X2predict.cols() != feature_dimension_occu_change_rate) {
        std::cerr << "Error: Input dimension mismatch. Expected " 
                  << feature_dimension_occu_change_rate << " columns, got " 
                  << X2predict.cols() << std::endl;
        return Eigen::VectorXd::Zero(X2predict.rows());
    }

    // Normalize input data
    Eigen::MatrixXd X_normalized = 2.0 * ((X2predict.rowwise() - X_min_occu_change_rate.transpose()).array().rowwise() / X_range_occu_change_rate.transpose().array()) - 1.0;
    
    // Call CUDA acceleration predict function
    Eigen::VectorXd Y_Predicted(X2predict.rows());
    
    try {
        fast_planner::predict_occupancy_change_rate_cuda(
            X_normalized.data(),
            theta.data(),
            randomWeights_occu.data(),
            randomBias_occu.data(),
            Y_Predicted.data(),
            X2predict.rows(),
            feature_dimension_occu_change_rate,
            targetDimen,
            actiFunc.c_str()
        );

    } catch (const std::exception& e) {
        std::cerr << "CUDA prediction failed: " << e.what() << std::endl;
        return Eigen::VectorXd::Zero(X2predict.rows());
    }

    // Apply threshold for classification problem
    Y_Predicted = (Y_Predicted.array() > 0.5).cast<double>();
    
    // // If ground truth is available (e.g., during testing)
    // if (Y_test_data_.size() > 0) {
    //     double test_score = score_occupancy_change_rate(X2predict, Y_test_data_);
    //     std::cout << "Test accuracy for occupancy change rate: " << test_score << std::endl;
    // }

    return Y_Predicted;
}

// This function calculates the distance field and fills it into ESDF
template <typename F_get_val, typename F_set_val>
void SDFMap::fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim) {
  int v[mp_->map_voxel_num_(dim)];
  double z[mp_->map_voxel_num_(dim) + 1];

  int k = start;
  v[start] = start;
  z[start] = -std::numeric_limits<double>::max();
  z[start + 1] = std::numeric_limits<double>::max();

  for (int q = start + 1; q <= end; q++) {
    k++;
    double s;

    do {
      k--;
      s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
    } while (s <= z[k]);

    k++;

    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::max();
  }

  k = start;

  for (int q = start; q <= end; q++) {
    while (z[k + 1] < q)
      k++;
    double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
    f_set_val(q, val);
  }
}

// This is a matrix representing the random weights in the model. These weights are used to map low-dimensional data (such as 3D coordinates) to a higher-dimensional space.
    Eigen::MatrixXd randomWeights;
// This is a matrix representing the random bias terms in the model, used together with randomWeights for feature mapping.
    Eigen::MatrixXd randomBias;
// This is a vector representing the model's parameter vector, which is usually obtained after model training.
    Eigen::VectorXd theta1;

void SDFMap::updateESDF3d() {
  Eigen::Vector3i min_esdf = md_->local_bound_min_;
  Eigen::Vector3i max_esdf = md_->local_bound_max_;

  if (mp_->optimistic_) {
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
      for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
        fillESDF(
            [&](int z) {
              return md_->occupancy_buffer_inflate_[toAddress(x, y, z)] == 1 ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
            max_esdf[2], 2);
      }
  } else {
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
      for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
        fillESDF(
            [&](int z) {
              int adr = toAddress(x, y, z);
              return (md_->occupancy_buffer_inflate_[adr] == 1 ||
                      md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3) ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
            max_esdf[2], 2);
      }
  }

  for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF(
          [&](int y) { return md_->tmp_buffer1_[toAddress(x, y, z)]; },
          [&](int y, double val) { md_->tmp_buffer2_[toAddress(x, y, z)] = val; }, min_esdf[1],
          max_esdf[1], 1);
    }
  for (int y = min_esdf[1]; y <= max_esdf[1]; y++)
    for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
      fillESDF(
          [&](int x) { return md_->tmp_buffer2_[toAddress(x, y, z)]; },
          [&](int x, double val) {
            md_->distance_buffer_[toAddress(x, y, z)] = mp_->resolution_ * std::sqrt(val);
          },
          min_esdf[0], max_esdf[0], 0);
    }

  if (mp_->signed_dist_) {
    // Compute negative distance
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
      for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
        fillESDF(
            [&](int z) {
              return md_->occupancy_buffer_inflate_
                          [x * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2) +
                           y * mp_->map_voxel_num_(2) + z] == 0 ?
                  0 :
                  std::numeric_limits<double>::max();
            },
            [&](int z, double val) { md_->tmp_buffer1_[toAddress(x, y, z)] = val; }, min_esdf[2],
            max_esdf[2], 2);
      }
    for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
      for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
        fillESDF(
            [&](int y) { return md_->tmp_buffer1_[toAddress(x, y, z)]; },
            [&](int y, double val) { md_->tmp_buffer2_[toAddress(x, y, z)] = val; }, min_esdf[1],
            max_esdf[1], 1);
      }
    for (int y = min_esdf[1]; y <= max_esdf[1]; y++)
      for (int z = min_esdf[2]; z <= max_esdf[2]; z++) {
        fillESDF(
            [&](int x) { return md_->tmp_buffer2_[toAddress(x, y, z)]; },
            [&](int x, double val) {
              md_->distance_buffer_neg_[toAddress(x, y, z)] = mp_->resolution_ * std::sqrt(val);
            },
            min_esdf[0], max_esdf[0], 0);
      }
    // Merge negative distance with positive
    for (int x = min_esdf(0); x <= max_esdf(0); ++x)
      for (int y = min_esdf(1); y <= max_esdf(1); ++y)
        for (int z = min_esdf(2); z <= max_esdf(2); ++z) {
          int idx = toAddress(x, y, z);
          if (md_->distance_buffer_neg_[idx] > 0.0)
            md_->distance_buffer_[idx] += (-md_->distance_buffer_neg_[idx] + mp_->resolution_);
        }
  }
}

int num_ex=0;
int num_ex_black = 0;
Eigen::MatrixXd point_local_map(52000,2);
Eigen::MatrixXd point_local_map_black(918*12,2);
Eigen::MatrixXd point_p(52000,3);
Eigen::MatrixXd point_p_black(918*12, 3); 
Eigen::VectorXd accuracy_Y(300);//ues to calculate accuracy
Eigen::Vector3d pt_w1;
Eigen::Vector3d pt_w1_black;
Eigen::MatrixXd point_p_black1(918*12, 3); 

int death = 0;
int acc_i=0;//use to calculate accuracy

void SDFMap::setCacheOccupancy(const int& adr, const int& occ) {
  // Add to update list if first visited
  if (md_->count_hit_[adr] == 0 && md_->count_miss_[adr] == 0){
    // if(death==0){  
    //   point_local_map(num_ex,0)=adr;
    //   point_local_map(num_ex,1)=occ;
    //   point_p(num_ex,0)=pt_w1[0];
    //   point_p(num_ex,1)=pt_w1[1];
    //   point_p(num_ex,2)=pt_w1[2];
    //   num_ex++;
    // }
    md_->cache_voxel_.push(adr);
  }
  if (occ == 0){
    md_->count_miss_[adr] = 1;
    //md_->count_hit_[adr] = 0;
  }
  else if (occ == 1){
    md_->count_hit_[adr] = 1;
    //md_->count_miss_[adr] = 0;
  }
}

void SDFMap::trainAndPredictOccupancy(
    std::vector<SimpleTrainingData>& training_buffer,
    const pcl::PointCloud<pcl::PointXYZ>& map_surface_value,
    random_mapping_method& rmm) {
      // std::cout << "rmm.theta before theta value: " << rmm.theta << std::endl;
    
    // Training stage
    if (!training_buffer.empty()) {
        size_t data_size = training_buffer.size();
        Eigen::MatrixXd new_point_p(data_size, 3);
        Eigen::MatrixXd new_point_local_map(data_size, 2);
        // new_point_p.resize(data_size, 3);
        // new_point_local_map.resize(data_size, 2);
        
        // Prepare visualization point cloud
        pcl::PointCloud<pcl::PointXYZ> training_cloud;
        training_cloud.reserve(data_size);
        
        // Process training data
        for (size_t i = 0; i < data_size; ++i) {
            const auto& data = training_buffer[i];
            new_point_p.row(i) = data.pos;
            new_point_local_map(i, 1) = data.occupancy;
            
            training_cloud.push_back(pcl::PointXYZ(
                data.pos.x(), data.pos.y(), data.pos.z()));
        }
        
        // Publish training points
        // publishPointCloud(training_cloud, training_points_pub);
        
        // Train model 
        rmm.Y = new_point_local_map.col(1);
        rmm.L_training = static_cast<int>(data_size);
        // std::cout << "rmm.L_training: " << rmm.L_training << std::endl;
        rmm.data_transformed = rmm.feature_mapping(new_point_p);
        // std::cout << "rmm.data_transformed: " << rmm.data_transformed << std::endl;
        rmm.X_training = rmm.data_transformed;
        rmm.Y_training = rmm.Y;
        rmm.train_occupancy_change_rate(rmm.X_training, rmm.Y_training, 1);//GTX 1650 optimization: reduced from 3 to 1
        // std::cout << "rmm.train before occupancy theta value: " << rmm.theta << std::endl;
        // rmm.train(rmm.X_training, rmm.Y_training, 3);
        
        training_buffer.clear();
        training_buffer.shrink_to_fit();
    }
    
    // Process grid points, point cloud downsampling
    std::unordered_map<int, int> grid_point_count;
    std::set<int> unique_adrs;
    processGridPoints(map_surface_value, grid_point_count, unique_adrs);
    
    // Prepare prediction data
    Eigen::MatrixXd point_p_black1(unique_adrs.size(), 3);
    int data_count = 0;
    
    // Process unique grid
    for (int adr : unique_adrs) {
        // Eigen::Vector3d pos;
        // Eigen::Vector3i idx = addressToIndex(adr);
        // indexToPos(idx, pos);

        int idx_x = adr % mp_->map_voxel_num_[0];
        int idx_y = (adr / mp_->map_voxel_num_[0]) % mp_->map_voxel_num_[1];
        int idx_z = adr / (mp_->map_voxel_num_[0] * mp_->map_voxel_num_[1]);
        Eigen::Vector3i idx(idx_x, idx_y, idx_z);

        Eigen::Vector3d pos;
        indexToPos(idx, pos);
        // std::cout << "pos: " << pos <<std::endl;
        // pos:   8.85
        // -19.25
        // -0.65
        // pos:   9.05
        // -19.25
        // -0.65
        // publishGridPoint(pos);
        point_p_black1.row(data_count++) = pos;
    }
    
    // Predict occupancy state
    if (point_p_black1.rows() > 0)
    {
        rmm.data_transformed_black = rmm.feature_mapping(point_p_black1);
        // std::cout << "rmm.data_transformed_black.cols: " << rmm.data_transformed_black.cols() << std::endl;
        
        Eigen::VectorXd Y_test_predicted_black = rmm.predict_occupancy_change_rate(rmm.data_transformed_black);
        // rmm.print_prediction_occu_stats(Y_test_predicted_black);
        // std::cout << "Y_test_predicted_black: " << Y_test_predicted_black << std::endl;
        
        // Visualize prediction results
        // publishPredictions(Y_test_predicted_black, point_p_black1);
        publishPredictions_new(Y_test_predicted_black, point_p_black1);
    }
}

void SDFMap::updateOccupancyChangeRateInfo() {
    try {
        // GTX 1650 optimization: training skip frame strategy
        training_frame_counter_++;
        
        // Scheme 1: train every N frames, but predict every frame
        if (training_frame_counter_ >= TRAIN_EVERY_N_FRAMES) {
            training_frame_counter_ = 0;  // Reset counter
            
            // std::cout << "rmm.theta before theta value: " << rmm.theta << std::endl;//step1: 
            // Train and predict occupancy state
            trainAndPredictOccupancy(training_buffer, map_surface_value, rmm);
            
            // Print training information
            // ROS_INFO_THROTTLE(5.0, "RMM training executed (every %d frames)", TRAIN_EVERY_N_FRAMES);
        } else {
            // Skip training, only predict
        }

        // // Safely update gradients
        // if (rmm.data_transformed.rows() > 0) {
        //     safeUpdateGradients(rmm.data_transformed, rmm);
        // }
    } catch (const std::exception& e) {
        ROS_ERROR("Error in occupancy update: %s", e.what());
    }
}

void SDFMap::publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud, 
                             ros::Publisher& publisher) {
    if (!mr_) return;
    
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::PointCloud<pcl::PointXYZ> cloud_copy = cloud;
    cloud_copy.header.frame_id = mr_->frame_id_;
    pcl::toROSMsg(cloud_copy, cloud_msg);
    publisher.publish(cloud_msg);
}

void SDFMap::publishGridPoint(const Eigen::Vector3d& pos) {
    if (!mr_) return;
    
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PointXYZ pt(pos.x(), pos.y(), pos.z());
    cloud.points.push_back(pt);
    cloud.width = 1;
    cloud.height = 1;
    cloud.is_dense = true;
    
    publishPointCloud(cloud, training_points_pub);
}

void SDFMap::publishPredictions(const Eigen::VectorXd& predictions,
                              const Eigen::MatrixXd& positions) {
    if (!mr_) return;
    
    pcl::PointCloud<pcl::PointXYZ> predic_cloud;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        Eigen::Vector3i idx;
        Eigen::Vector3d pos = positions.row(i);
        posToIndex(pos, idx);
        
        md_->occupancy_buffer_inflate_[toAddress(idx)] = predictions(i);
        
        if (predictions(i) == 1) {
            predic_cloud.push_back(pcl::PointXYZ(
                positions(i, 0), positions(i, 1), positions(i, 2)));
        }
    }
    
    publishPointCloud(predic_cloud, mr_->map_pub_predict);
}

void SDFMap::publishPredictions_new(const Eigen::VectorXd& predictions,
                              const Eigen::MatrixXd& positions) {
    if (!mr_) return;
    
    pcl::PointCloud<pcl::PointXYZ> predic_cloud;
    
    // Set sampling interval and occupancy threshold
    const int sample_interval = 2;  
    const double occupancy_threshold = 0.8;  // Occupancy probability threshold
    
    try {
        for (size_t i = 0; i < predictions.size(); i += sample_interval) {
            // Safe check
            if (i >= positions.rows()) break;
            
            Eigen::Vector3i idx;
            Eigen::Vector3d pos = positions.row(i);
            
            // Check position validity
            if (!isInMap(pos)) continue;
            
            posToIndex(pos, idx);
            int adr = toAddress(idx);
            
            // Check address validity
            if (adr < 0 || adr >= md_->occupancy_buffer_inflate_.size()) continue;
            
            // Use threshold to determine occupancy
            if (predictions(i) >= occupancy_threshold) {
                md_->occupancy_buffer_inflate_[adr] = 1;
                predic_cloud.push_back(pcl::PointXYZ(pos(0), pos(1), pos(2)));
            } else {
                md_->occupancy_buffer_inflate_[adr] = 0;
            }
        }
        
        // Perform voxel grid downsampling on the point cloud
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        vg.setInputCloud(predic_cloud.makeShared());
        vg.setLeafSize(0.1f, 0.1f, 0.1f);  // Set voxel grid leaf size
        vg.filter(*cloud_filtered);
        
        // Publish downsampled point cloud
        publishPointCloud(*cloud_filtered, mr_->map_pub_predict);
        
    } catch (const std::exception& e) {
        ROS_ERROR("Error in publishPredictions: %s", e.what());
    }
}

void SDFMap::inputPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>& points,// points is the original point cloud data passed in
    const int& point_num,
    const Eigen::Vector3d& camera_pos) {
  if (point_num == 0) return;
  md_->raycast_num_ += 1;

  auto ti=ros::Time::now();

  //  ROS_WARN("buffer_size");

  Eigen::Vector3d update_min = camera_pos;
  Eigen::Vector3d update_max = camera_pos;
  if (md_->reset_updated_box_) {
    md_->update_min_ = camera_pos;
    md_->update_max_ = camera_pos;
    md_->reset_updated_box_ = false;
  }

  // Eigen::Vector3d pt_w, tmp;
  Eigen::Vector3d pt_w, tmp, pt_w_black; 

  Eigen::Vector3i idx, idx_black;  
  int vox_adr, vox_adr_black;  // Voxel address
  // Eigen::Vector3i idx;
  // int vox_adr;
  double length;

  for (int i = 0; i < point_num; ++i) {
    auto& pt = points.points[i];
    pt_w << pt.x, pt.y, pt.z;


    int tmp_flag;
    // Set flag for projected point
    if (!isInMap(pt_w)) {
      // Find closest point in map and set free
      pt_w = closetPointInMap(pt_w, camera_pos);
      length = (pt_w - camera_pos).norm();
      if (length > mp_->max_ray_length_)
        pt_w = (pt_w - camera_pos) / length * mp_->max_ray_length_ + camera_pos;
      if (pt_w[2] < 0.2){
        continue;
      }
      tmp_flag = 0;
    } else {
      length = (pt_w - camera_pos).norm();
      if (length > mp_->max_ray_length_) {
        pt_w = (pt_w - camera_pos) / length * mp_->max_ray_length_ + camera_pos;
      if (pt_w[2] < 0.2){
        continue;
      }
        tmp_flag = 0;
      } else
        tmp_flag = 1;
    }
    pt_w1=pt_w;
    posToIndex(pt_w, idx); // Convert position to index
    vox_adr = toAddress(idx); // Calculate address corresponding to index
    setCacheOccupancy(vox_adr, tmp_flag);
    death=0;

    for (int k = 0; k < 3; ++k) {
      update_min[k] = min(update_min[k], pt_w[k]);
      update_max[k] = max(update_max[k], pt_w[k]);
    }
    // Raycasting between camera center and point
    if (md_->flag_rayend_[vox_adr] == md_->raycast_num_)
      continue;
    else
      md_->flag_rayend_[vox_adr] = md_->raycast_num_;

    caster_->input(pt_w, camera_pos);
    caster_->nextId(idx);
    while (caster_->nextId(idx)){
      death=1;
      setCacheOccupancy(toAddress(idx), 0);
    }
    death=0;
  }

  Eigen::Vector3d bound_inf(mp_->local_bound_inflate_, mp_->local_bound_inflate_, 0);
  
// Convert the positions of `update_max` and `update_min` to the index of the grid map, and update the boundary of the map
  posToIndex(update_max + bound_inf, md_->local_bound_max_);
  posToIndex(update_min - bound_inf, md_->local_bound_min_);
  boundIndex(md_->local_bound_min_);
  boundIndex(md_->local_bound_max_);

// Set the local update flag to true, indicating that the map has been updated
  mr_->local_updated_ = true;

  // Bounding box for subsequent updating
  for (int k = 0; k < 3; ++k) {
    md_->update_min_[k] = min(update_min[k], md_->update_min_[k]);
    md_->update_max_[k] = max(update_max[k], md_->update_max_[k]);
  }
  //  ROS_WARN("cache=%d",md_->cache_voxel_.size());

  while (!md_->cache_voxel_.empty()) {
    int adr = md_->cache_voxel_.front();
    md_->cache_voxel_.pop();
    double log_odds_update =
        md_->count_hit_[adr] >= md_->count_miss_[adr] ? mp_->prob_hit_log_ : mp_->prob_miss_log_;
    md_->count_hit_[adr] = md_->count_miss_[adr] = 0;
    if (md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3)
      md_->occupancy_buffer_[adr] = mp_->min_occupancy_log_;

    md_->occupancy_buffer_[adr] = std::min(
        std::max(md_->occupancy_buffer_[adr]+log_odds_update+0.33, mp_->clamp_min_log_),
        mp_->clamp_max_log_);


    // Convert log odds to probability, then binarize
    double prob = 1.0 / (1.0 + exp(-md_->occupancy_buffer_[adr]));
    int binary_occupancy = (prob > 0.5) ? 1 : 0;  // Use 0.5 as threshold
    // std::cout << "Log odds value: " << md_.occupancy_buffer_[adr] << std::endl;// Original log odds value
    // std::cout << "Probability: " << prob << std::endl;// Probability
    // std::cout << "Binary occupancy: " << binary_occupancy << std::endl;// Binarization
    // std::cout << "adr: " << adr << std::endl;
    // When storing training data, use simplified logic to collect training data including occupancy states of 0 and 1
    if (std::abs(md_->occupancy_buffer_[adr]) > 1e-6) {
        // Only add new data when the buffer is not full
        if (training_buffer.size() < MAX_TRAINING_SIZE) {
          // std::cout << "training_buffer.size()=" << training_buffer.size() << std::endl;
            // Convert address to voxel index
            int x, y, z;
            // addressToIndex(adr, x, y, z);
            // x = adr % mp_.map_voxel_num_[0];
            // y = (adr / mp_.map_voxel_num_[0]) % mp_.map_voxel_num_[1];
            // z = adr / (mp_.map_voxel_num_[0] * mp_.map_voxel_num_[1]);
            addressToIndex_occu_change_rate_training_buffer(adr, x, y, z);
            Eigen::Vector3i voxel(x, y, z);

            // Convert voxel index to actual position
            Eigen::Vector3d pos;
            indexToPos(voxel, pos);
            
            double prob = 1.0 / (1.0 + exp(-md_->occupancy_buffer_[adr]));
            int binary_occupancy = (prob > 0.5) ? 1 : 0;
            
            training_buffer.push_back({pos, binary_occupancy});
        }
    }
  }

  // Occupancy update part of the map
  updateOccupancyChangeRateInfo();

  num_ex=0;
double grad_map_time =  (ros::Time::now() - ti).toSec();
}

Eigen::Vector3d
SDFMap::closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt) {
  Eigen::Vector3d diff = pt - camera_pt;
  Eigen::Vector3d max_tc = mp_->map_max_boundary_ - camera_pt;
  Eigen::Vector3d min_tc = mp_->map_min_boundary_ - camera_pt;
  double min_t = 1000000;
  for (int i = 0; i < 3; ++i) {
    if (fabs(diff[i]) > 0) {
      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t) min_t = t1;
      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t) min_t = t2;
    }
  }
  return camera_pt + (min_t - 1e-3) * diff;
}

void SDFMap::clearAndInflateLocalMap() {
  // update inflated occupied cells
  // clean outdated occupancy
  int inf_step = ceil(mp_->obstacles_inflation_ / mp_->resolution_);
  vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));
  // inf_pts.resize(4 * inf_step + 3);

  for (int x = md_->local_bound_min_(0); x <= md_->local_bound_max_(0); ++x)
    for (int y = md_->local_bound_min_(1); y <= md_->local_bound_max_(1); ++y)
      for (int z = md_->local_bound_min_(2); z <= md_->local_bound_max_(2); ++z) {
        md_->occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
      }

  // inflate newest occpuied cells
  for (int x = md_->local_bound_min_(0); x <= md_->local_bound_max_(0); ++x)
    for (int y = md_->local_bound_min_(1); y <= md_->local_bound_max_(1); ++y)
      for (int z = md_->local_bound_min_(2); z <= md_->local_bound_max_(2); ++z) {
        int id1 = toAddress(x, y, z);
        if (md_->occupancy_buffer_[id1] > mp_->min_occupancy_log_) {
          inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);

          for (auto inf_pt : inf_pts) {
            int idx_inf = toAddress(inf_pt);
            if (idx_inf >= 0 &&
                idx_inf <
                    mp_->map_voxel_num_(0) * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2)) {
              md_->occupancy_buffer_inflate_[idx_inf] = 1;
            }
          }
        }
      }

  // add virtual ceiling to limit flight height
  if (mp_->virtual_ceil_height_ > -0.5) {
    int ceil_id = floor((mp_->virtual_ceil_height_ - mp_->map_origin_(2)) * mp_->resolution_inv_);
    for (int x = md_->local_bound_min_(0); x <= md_->local_bound_max_(0); ++x)
      for (int y = md_->local_bound_min_(1); y <= md_->local_bound_max_(1); ++y) {
        // md_->occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;
        md_->occupancy_buffer_[toAddress(x, y, ceil_id)] = mp_->clamp_max_log_;
      }
  }
}

double SDFMap::getResolution() {
  return mp_->resolution_;
}

int SDFMap::getVoxelNum() {
  return mp_->map_voxel_num_[0] * mp_->map_voxel_num_[1] * mp_->map_voxel_num_[2];
}

void SDFMap::getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size) {
  ori = mp_->map_origin_, size = mp_->map_size_;
}

void SDFMap::getBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax) {
  bmin = mp_->box_mind_;
  bmax = mp_->box_maxd_;
}

void SDFMap::getUpdatedBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax, bool reset) {
  // Get the current updated bounding box, `bmin` is the minimum boundary, `bmax` is the maximum boundary
  bmin = md_->update_min_;  // Update minimum boundary
  bmax = md_->update_max_;  // Update maximum boundary
  // If `reset` flag is true, set `reset_updated_box_` to true
  // indicating that the boundary has been reset, used for next update
  if (reset) md_->reset_updated_box_ = true;
}

double SDFMap::getDistWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad) {
  if (!isInMap(pos)) {
    grad.setZero();
    return 0;
  }

  /* trilinear interpolation */
  Eigen::Vector3d pos_m = pos - 0.5 * mp_->resolution_ * Eigen::Vector3d::Ones();
  Eigen::Vector3i idx;
  posToIndex(pos_m, idx);
  Eigen::Vector3d idx_pos, diff;
  indexToPos(idx, idx_pos);
  diff = (pos - idx_pos) * mp_->resolution_inv_;

  double values[2][2][2];
  for (int x = 0; x < 2; x++)
    for (int y = 0; y < 2; y++)
      for (int z = 0; z < 2; z++) {
        Eigen::Vector3i current_idx = idx + Eigen::Vector3i(x, y, z);
        values[x][y][z] = getDistance(current_idx);
      }

  double v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
  double v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
  double v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
  double v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];
  double v0 = (1 - diff[1]) * v00 + diff[1] * v10;
  double v1 = (1 - diff[1]) * v01 + diff[1] * v11;
  double dist = (1 - diff[2]) * v0 + diff[2] * v1;

  grad[2] = (v1 - v0) * mp_->resolution_inv_;
  grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) * mp_->resolution_inv_;
  grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
  grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
  grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
  grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);
  grad[0] *= mp_->resolution_inv_;

  return dist;
}
}  // namespace fast_planner

// SDFMap