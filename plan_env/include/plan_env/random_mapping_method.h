#ifndef _RANDOM_MAPPING_METHOD_H
#define _RANDOM_MAPPING_METHOD_H

#include <Eigen/Core>
#include <string>
#include <cmath>
#include <iostream> 
#include <Eigen/Eigen>

namespace fast_planner {

class random_mapping_method {
public:

    // Get the trained parameters
    const Eigen::MatrixXd& getRandomWeights() const { return randomWeights_occu; }
    const Eigen::MatrixXd& getRandomBias() const { return randomBias_occu; }
    const Eigen::VectorXd& getMinOccu() const { return X_min_occu_change_rate; }
    const Eigen::VectorXd& getMaxOccu() const { return X_max_occu; }

    // Use existing parameters for feature mapping
    Eigen::MatrixXd feature_mapping_with_params(
        Eigen::MatrixXd& dataSet,
        const Eigen::MatrixXd& weights,
        const Eigen::MatrixXd& bias);

    // Common members
    Eigen::MatrixXd X_training;
    Eigen::VectorXd Y_training;
    Eigen::MatrixXd data_transformed;
    Eigen::MatrixXd data_transformed_black;
    Eigen::MatrixXd randomBias_occu_change_rate;
    // Eigen::MatrixXd randomWeights_occu_change_rate;
    Eigen::VectorXd theta; 
    Eigen:: VectorXd Y;
    int L_training;
    // Occupancy change rate training related members
    Eigen::MatrixXd X_training_data_occu_change_rate;
    Eigen::VectorXd Y_training_data_occu_change_rate;
    Eigen::MatrixXd X_normalized_occu_change_rate;
    Eigen::VectorXd Y_normalized_occu_change_rate;
    int training_samples_occu_change_rate;
    int feature_dimension_occu_change_rate;
    Eigen::VectorXd X_min_occu_change_rate;
    Eigen::VectorXd X_range_occu_change_rate;
    double Y_min_occu_change_rate;
    double Y_max_occu_change_rate;
    double Y_range_occu_change_rate;
    Eigen::MatrixXd data_transformed_kinodynamic;
    Eigen::MatrixXd data_transformed_kinodynamic_expand;

    Eigen::VectorXd X_min_;
    Eigen::VectorXd X_max_;
    Eigen::VectorXd X_range_;
    double Y_min_;
    double Y_range_;

    // Add training related common members
    Eigen::VectorXd Y_training_data_;    // Original training data Y
    Eigen::MatrixXd X_training_data_;    // Original training data X
    Eigen::VectorXd Y_normalized_;       // Normalized Y
    Eigen::MatrixXd X_normalized_;       // Normalized X
    double Y_max_;                       // Maximum value of Y
    int training_samples_;               // Training sample size
    int feature_dimension_;              // Feature dimension

    // Maximum and minimum normalization, occupancy
    Eigen::VectorXd X_max_occu;
    // Initialize random weights and bias, occupancy
    Eigen::MatrixXd randomWeights_occu;
    Eigen::MatrixXd randomBias_occu;

    // Maximum and minimum normalization
    Eigen::VectorXd denormalize(const Eigen::VectorXd& y) const {
        return ((y.array() + 1.0) * Y_range_ / 2.0) + Y_min_;
    }

    // Use minimum-maximum normalization normalize function
    Eigen::VectorXd normalize_input(const Eigen::VectorXd& x) const {
        return 2.0 * ((x - X_min_).array() / X_range_.array()) - 1.0;
    }
    void print_prediction_stats(const Eigen::VectorXd& predictions) {
    // Denormalize predictions
    // Eigen::VectorXd denormalized_predictions = ((predictions.array() + 1.0) * Y_range_ / 2.0) + Y_min_;
    Eigen::VectorXd denormalized_predictions = predictions;
    
    // Calculate statistical of predicted data
    double pred_min = denormalized_predictions.minCoeff();
    double pred_max = denormalized_predictions.maxCoeff();
    double pred_mean = denormalized_predictions.mean();
    double pred_std = sqrt((denormalized_predictions.array() - pred_mean).square().sum() / 
        (denormalized_predictions.size() - 1));
    double pred_range = pred_max - pred_min;
    
    // Use stored training data to calculate statistical
    double train_min = Y_training_data_.minCoeff();
    double train_max = Y_training_data_.maxCoeff();
    double train_mean = Y_training_data_.mean();
    double train_std = sqrt((Y_training_data_.array() - train_mean).square().sum() / 
        (Y_training_data_.size() - 1));
    double train_range = train_max - train_min;
    
    // printf("\n=== elevation Statistical Analysis ===\n");
    // printf("\nelevationPrediction Statistics:\n");
    // printf("Min height:     %.6f\n", pred_min);
    // printf("Max height:     %.6f\n", pred_max);
    // printf("Mean height:    %.6f\n", pred_mean);
    // printf("Std deviation:  %.6f\n", pred_std);
    // printf("Range:         %.6f\n", pred_range);
    
    // printf("\nelevation Training Data Statistics:\n");
    // printf("Min height:     %.6f\n", train_min);
    // printf("Max height:     %.6f\n", train_max);
    // printf("Mean height:    %.6f\n", train_mean);
    // printf("Std deviation:  %.6f\n", train_std);
    // printf("Range:         %.6f\n", train_range);
    
    // printf("\nelevation Comparison Metrics:\n");
    // printf("Mean difference:     %.6f\n", std::abs(pred_mean - train_mean));
    // printf("Std dev difference:  %.6f\n", std::abs(pred_std - train_std));
    // printf("Range difference:    %.6f\n", std::abs(pred_range - train_range));
    
    // Calculate the proportion of predicted values within the training data range
    int within_range = 0;
    for(int i = 0; i < denormalized_predictions.size(); i++) {
        if(denormalized_predictions(i) >= train_min && 
           denormalized_predictions(i) <= train_max) {
            within_range++;
        }
    }
    double within_range_ratio = (double)within_range / denormalized_predictions.size() * 100.0;
    printf("\nPredictions within training range: %.2f%%\n", within_range_ratio);
    
    // Calculate distribution statistics
    // int within_1_std = 0, within_2_std = 0, within_3_std = 0;
    // for(int i = 0; i < denormalized_predictions.size(); i++) {
    //     double z_score = (denormalized_predictions(i) - train_mean) / train_std;
    //     if(std::abs(z_score) <= 1.0) within_1_std++;
    //     if(std::abs(z_score) <= 2.0) within_2_std++;
    //     if(std::abs(z_score) <= 3.0) within_3_std++;
    // }
    
    // printf("\nDistribution Analysis:\n");
    // printf("Within ±1σ (68.27%% expected): %.2f%%\n", 
    //        (double)within_1_std / denormalized_predictions.size() * 100.0);
    // printf("Within ±2σ (95.45%% expected): %.2f%%\n", 
    //        (double)within_2_std / denormalized_predictions.size() * 100.0);
    // printf("Within ±3σ (99.73%% expected): %.2f%%\n", 
    //        (double)within_3_std / denormalized_predictions.size() * 100.0);
}

void print_prediction_occu_stats(const Eigen::VectorXd& predictions) {
    // Check if the input is empty
    if (predictions.size() == 0) {
        printf("Error: Predictions vector is empty!\n");
        return;
    }

    // Check if the training data is empty
    if (Y_training_data_occu_change_rate.size() == 0) {
        printf("Error: Training data is empty! Please ensure the model is trained first.\n");
        return;
    }

    // Use original predictions
    Eigen::VectorXd denormalized_predictions = predictions;
    
    // Calculate statistical of predicted data
    double pred_min = denormalized_predictions.minCoeff();
    double pred_max = denormalized_predictions.maxCoeff();
    double pred_mean = denormalized_predictions.mean();
    double pred_std = sqrt((denormalized_predictions.array() - pred_mean).square().sum() / 
        (denormalized_predictions.size() - 1));
    double pred_range = pred_max - pred_min;
    
    // Use stored training data to calculate statistical
    double train_min = Y_training_data_occu_change_rate.minCoeff();
    double train_max = Y_training_data_occu_change_rate.maxCoeff();
    double train_mean = Y_training_data_occu_change_rate.mean();
    double train_std = sqrt((Y_training_data_occu_change_rate.array() - train_mean).square().sum() / 
        (Y_training_data_occu_change_rate.size() - 1));
    double train_range = train_max - train_min;
    
    printf("\n=== Occupancy Change Rate Statistical Analysis ===\n");
    printf("\noccu Prediction Statistics:\n");
    printf("Min rate:      %.6f\n", pred_min);
    printf("Max rate:      %.6f\n", pred_max);
    printf("Mean rate:     %.6f\n", pred_mean);
    printf("Std deviation: %.6f\n", pred_std);
    printf("Range:         %.6f\n", pred_range);
    
    printf("\noccu Training Data Statistics:\n");
    printf("Min rate:      %.6f\n", train_min);
    printf("Max rate:      %.6f\n", train_max);
    printf("Mean rate:     %.6f\n", train_mean);
    printf("Std deviation: %.6f\n", train_std);
    printf("Range:         %.6f\n", train_range);
    
    printf("\noccuComparison Metrics:\n");
    printf("Mean difference:     %.6f\n", std::abs(pred_mean - train_mean));
    printf("Std dev difference:  %.6f\n", std::abs(pred_std - train_std));
    printf("Range difference:    %.6f\n", std::abs(pred_range - train_range));
    
    // Calculate the proportion of predicted values within the training data range
    int within_range = 0;
    for(int i = 0; i < denormalized_predictions.size(); i++) {
        if(denormalized_predictions(i) >= train_min && 
           denormalized_predictions(i) <= train_max) {
            within_range++;
        }
    }
    double within_range_ratio = (double)within_range / denormalized_predictions.size() * 100.0;
    printf("\nPredictions within training range: %.2f%%\n", within_range_ratio);
}
    // Use constructor with default parameters (for GTX 1650 aggressive optimization)
    random_mapping_method(int targetDimen = 16,  // Significantly reduced from 50 to 16 to decrease computation
                         std::string actiFunc = "sin", 
                         double scaleRate = 3.2, 
                         std::string learnerType = "cla")
        : actiFunc(actiFunc), 
          targetDimen(targetDimen), 
          scaleRate(scaleRate), 
          learnerType(learnerType) {
        theta = Eigen::VectorXd::Zero(targetDimen + 1);
    }

    // Common members
    Eigen::MatrixXd generateRandomWeights1(int input_dim, int output_dim, double scaleRate);
    Eigen::MatrixXd generateRandomBias1(int input_dim, int output_dim, double scaleRate);
    // Eigen::MatrixXd feature_mapping1(Eigen::MatrixXd& dataSet, 
    //                                Eigen::MatrixXd& randomWeights, 
    //                                Eigen::MatrixXd& randomBias);
    Eigen::MatrixXd feature_mapping(Eigen::MatrixXd& dataSet);
    Eigen::MatrixXd feature_mapping_cpu(Eigen::MatrixXd& dataSet);

    void train1(Eigen::MatrixXd X, Eigen::VectorXd Y, int num_epochs);
    void train(Eigen::MatrixXd X, Eigen::VectorXd Y, int num_epochs);
    void train_occupancy_change_rate(Eigen::MatrixXd X, Eigen::VectorXd Y, int num_epochs);

    double sigmoid(double z);
    Eigen::VectorXd predict_occupancy_change_rate(const Eigen::MatrixXd& X2predict);
    Eigen::VectorXd predict_cpu(const Eigen::MatrixXd& X2predict);
    Eigen::MatrixXd computeGradients_cpu(const Eigen::MatrixXd& dataSet);
    Eigen::VectorXd predict_threshold(const Eigen::MatrixXd& X2predict, double threshold = 0.0);
    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X2predict);
    Eigen::VectorXd read_theta();
    double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y);

public:
    int targetDimen;
private:
    std::string actiFunc;
    // int targetDimen;
    double scaleRate;
    std::string learnerType;
    Eigen::VectorXd coef;
    int num_features;
    Eigen::VectorXd bias;
    Eigen::VectorXd weights;
    // Occupancy change rate learning rate
    double learning_rate_occu_change_rate = 0.1;

    // Private members
    Eigen::MatrixXd applyActivationFunction(const Eigen::MatrixXd& dataSet);
};
}  // namespace fast_planner
#endif // _RANDOM_MAPPING_METHOD_H