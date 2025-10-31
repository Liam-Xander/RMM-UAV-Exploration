
/*
 * @brief Implementation of the Random Mapping Method (RMM).
 *
 * This file contains the core logic for the RMM-based occupancy mapping approach.
 * It includes functions for:
 * 1. High-dimensional feature mapping using fixed random weights.
 * 2. Training the linear model (theta parameters) via gradient descent.
 * 3. Predicting occupancy probabilities for given 3D coordinates.
 *
 * The methods are designed to be lightweight and efficient, with support for
 * CUDA-accelerated computations, making them suitable for real-time
 * robotic applications.
 */

#include "plan_env/random_mapping_method.h"
#include <cmath>

namespace fast_planner {
Eigen::MatrixXd random_mapping_method::generateRandomWeights1(int input_dim, int output_dim, double scaleRate) {
    return Eigen::MatrixXd::Random(input_dim, output_dim) * scaleRate;
}

Eigen::MatrixXd random_mapping_method::generateRandomBias1(int input_dim, int output_dim, double scaleRate) {
    return Eigen::MatrixXd::Random(input_dim, output_dim) * scaleRate;
}

void random_mapping_method::train1(Eigen::MatrixXd X, Eigen::VectorXd Y, int num_epochs) {
    int m = X.rows();
    int n = X.cols();
    theta.setZero();  

    Eigen::MatrixXd X_bias(m, n + 1);
    X_bias << X, Eigen::MatrixXd::Ones(m, 1);

    for (int iter = 0; iter < num_epochs; ++iter) {
        Eigen::VectorXd predictions = X_bias * theta;
        Eigen::VectorXd h = predictions;
        double loss = ((h - Y).array().square()).sum() / (2 * m);
        Eigen::VectorXd error = h - Y;
        Eigen::VectorXd gradient = (X_bias.transpose() * error) / m;
        theta -= 0.16 * gradient;
    }
}

void random_mapping_method::train(Eigen::MatrixXd X, Eigen::VectorXd Y, int num_epochs) {
    int m = X.rows();
    int n = X.cols();
    Eigen::VectorXd error(m);

    Eigen::MatrixXd X_bias(m, n + 1);
    X_bias << X, Eigen::MatrixXd::Ones(m, 1);
    // std::cout << "train before occupancy theta value: " << theta << std::endl;
    for (int iter = 0; iter < num_epochs; ++iter) {
        Eigen::VectorXd predictions = X_bias * theta;
        Eigen::VectorXd h = predictions.array().unaryExpr([this](double x) { return sigmoid(x); });
        double loss = ((h - Y).array().square()).sum() / (2 * m);
        error = h - Y;
        Eigen::VectorXd gradient = (X_bias.transpose() * error) / m;
        theta -= learning_rate_occu_change_rate * gradient;
    }
    // std::cout << "trainafter occupancy theta value: " << theta << std::endl;
}

double random_mapping_method::sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

Eigen::VectorXd random_mapping_method::predict_threshold(const Eigen::MatrixXd& X2predict, double threshold) {
    Eigen::MatrixXd X2predict_augmented(X2predict.rows(), X2predict.cols() + 1);
    X2predict_augmented << X2predict, Eigen::MatrixXd::Constant(X2predict.rows(), 1, 1);
    
    Eigen::VectorXd Y_Predicted = X2predict_augmented * theta;
    
    Y_Predicted = Y_Predicted.array().unaryExpr([](double x) { 
        return 1.0 / (1.0 + exp(-x)); 
    });
    
    if(learnerType=="cla"){
        if (threshold < 0.3) {
            threshold = 0.5;
        }
        Y_Predicted = (Y_Predicted.array() > threshold).cast<double>();
    }
    
    return Y_Predicted;
}

Eigen::VectorXd random_mapping_method::predict_proba(const Eigen::MatrixXd& X2predict) {
    Eigen::MatrixXd X2predict_augmented(X2predict.rows(), X2predict.cols() + 1);
    X2predict_augmented << X2predict, Eigen::MatrixXd::Constant(X2predict.rows(), 1, 1);
    
    Eigen::VectorXd probs = X2predict_augmented * theta;
    probs = probs.array().unaryExpr([](double x) { 
        return 1.0 / (1.0 + exp(-x)); 
    });
    
    return probs;
}

Eigen::VectorXd random_mapping_method::read_theta() {
    return theta;
}

double random_mapping_method::score(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) {
    Eigen::VectorXd Y_Predicted = predict_occupancy_change_rate(X);
    int correct_predictions = (Y_Predicted.array() == Y.array()).count();
    double accuracy = static_cast<double>(correct_predictions) / Y.size();
    return accuracy;
}

// Activation function
Eigen::MatrixXd random_mapping_method::applyActivationFunction(const Eigen::MatrixXd& dataSet) {
    Eigen::MatrixXd result(dataSet.rows(), dataSet.cols());
    for (int i = 0; i < dataSet.rows(); ++i) {
        for (int j = 0; j < dataSet.cols(); ++j) {
            if (actiFunc == "sigmoid") {
                result(i, j) = 1.0 / (1 + exp(-dataSet(i, j)));
            }
            else if (actiFunc == "sin") {
                result(i, j) = sin(dataSet(i, j));
            }
            else if (actiFunc == "linear") {
                result(i, j) = dataSet(i, j);
            }
            else if (actiFunc == "tanh") {
                result(i, j) = tanh(dataSet(i, j));
            }
        }
    }
    return result;
}

// feature_mapping
Eigen::MatrixXd random_mapping_method::feature_mapping_cpu(Eigen::MatrixXd& dataSet) {
    int initial_dim = dataSet.cols();
    Eigen::MatrixXd randomWeights = Eigen::MatrixXd::Random(initial_dim, targetDimen) * scaleRate;
    Eigen::MatrixXd randomBias = Eigen::MatrixXd::Random(initial_dim, targetDimen) * scaleRate;
    Eigen::MatrixXd randomSetTemp = dataSet * randomWeights + Eigen::MatrixXd::Constant(dataSet.rows(), 3, 1) * randomBias;
    Eigen::MatrixXd randomSet = applyActivationFunction(randomSetTemp);
    return randomSet;
}

Eigen::VectorXd random_mapping_method::predict_cpu(const Eigen::MatrixXd& X2predict) {
    Eigen::MatrixXd X2predict_augmented(X2predict.rows(), X2predict.cols() + 1);
    X2predict_augmented << X2predict, Eigen::MatrixXd::Constant(X2predict.rows(), 1, 1);
    Eigen::VectorXd Y_Predicted =X2predict_augmented * theta;
    //Y_Predicted =Y_Predicted.array().unaryExpr([this](double x) { return sigmoid(x); });
    if(learnerType=="cla"){
         Y_Predicted = (Y_Predicted.array() >0).cast<double>();// Treat values greater than 0 as 1, indicating occupied; values less than or equal to 0 as 0, indicating free
    }
            
    return Y_Predicted;
}

// Calculate gradients
Eigen::MatrixXd random_mapping_method::computeGradients_cpu(const Eigen::MatrixXd& dataSet) {
    int m = dataSet.rows();
    int n = dataSet.cols();
    Eigen::MatrixXd gradients(m, 6);  // dx, dy, dz, dxx, dyy, dzz

    // Generate random weights and bias
    Eigen::MatrixXd randomWeights = generateRandomWeights1(n, targetDimen, scaleRate);
    Eigen::MatrixXd randomBias = generateRandomBias1(n, targetDimen, scaleRate);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < targetDimen; ++j) {
            double feature_val = dataSet(i, j);
            double wx = randomWeights(j, 0);
            double wy = randomWeights(j, 1);
            double wz = randomWeights(j, 2);
            double theta_val = theta(j);  

            // Calculate trigonometric terms
            double cos_term = cos(feature_val);
            double sin_term = -sin(feature_val);

            // Calculate gradient components
            double common = cos_term * theta_val;
            gradients(i, 0) += common * wx;  // dx
            gradients(i, 1) += common * wy;  // dy
            gradients(i, 2) += common * wz;  // dz

            // Calculate Hessian components
            common = sin_term * theta_val;
            gradients(i, 3) += common * wx * wx;  // dxx
            gradients(i, 4) += common * wy * wy;  // dyy
            gradients(i, 5) += common * wz * wz;  // dzz
        }
    }

    return gradients;
}
}  // namespace fast_planner
