#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  is_initialized_=false;

  // state dimension
  n_x_=5;

  // augmented state dimension
  n_aug_=7;

  // initialize lamdba
  lambda_=3-n_aug_;

  // intialize weights vector
  weights_=VectorXd(2*n_aug_+1);
  weights_(0)=lambda_/(lambda_+n_aug_);
  weights_.segment(1,2*n_aug_).fill(0.5/(lambda_+n_aug_));
  // weights as diagonal matrix
  wMat_=weights_.asDiagonal();

  // initialize Xsig_pred_
  Xsig_pred_=MatrixXd(n_x_,2*n_aug_+1);

  // initialize Xsig_aug_
  Xsig_aug_=MatrixXd(n_aug_,2*n_aug_+1);

  // initialize X Difference matrix XD_
  XD_=MatrixXd(n_x_,2*n_aug_+1);

  // set R_radar_
  R_radar_=MatrixXd(3,3);
  R_radar_ << std_radr_*std_radr_,0.0,0.0,
              0.0,std_radphi_*std_radphi_,0.0,
              0.0,0.0,std_radrd_*std_radrd_;

  // set R_lidar_
  R_lidar_=MatrixXd(2,2);
  R_lidar_ << std_laspx_*std_laspx_,0.0,
              0.0,std_laspy_*std_laspy_;
}

UKF::~UKF() {}

void UKF::NormalizeAngle(double& angle)
{
  if(angle>=M_PI) angle -= 2.0*M_PI;
  else if(angle<-M_PI) angle += 2.0*M_PI;
}

void UKF::ComputeAugmentedSigmaPoints()
{
  VectorXd x_aug(n_aug_);
  x_aug.segment(0,n_x_)=x_;
  x_aug.segment(n_x_,n_aug_-n_x_).fill(0.0);

  MatrixXd P_aug(n_aug_,n_aug_);
  P_aug.fill(0.0);
  P_aug.block(0,0,n_x_,n_x_)=P_;
  P_aug.block(n_x_,n_x_,n_aug_-n_x_,n_aug_-n_x_) << std_a_*std_a_,0.0,
                                                  0.0,std_yawdd_*std_yawdd_;

  MatrixXd P_aug_sqrt=P_aug.llt().matrixL();
  double spreadFactor=sqrt(lambda_+n_aug_);

  Xsig_aug_.col(0)=x_aug;
  for(int i=0;i<n_aug_;++i)
  {
    Xsig_aug_.col(i+1)=x_aug + spreadFactor*P_aug_sqrt.col(i);
    Xsig_aug_.col(i+1+n_aug_)=x_aug - spreadFactor*P_aug_sqrt.col(i);
  }
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if(!is_initialized_)
  {
    if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0.0, 0.0, 0.0;
      P_ << std_laspx_*std_laspx_,0.0,0.0,0.0,0.0,
            0.0,std_laspy_*std_laspy_,0.0,0.0,0.0,
            0.0,0.0,10.0,0.0,0.0,
            0.0,0.0,0.0,1.0,0.0,
            0.0,0.0,0.0,1.0,0.0;
    }
    else
    {
      double r=meas_package.raw_measurements_(0), theta=meas_package.raw_measurements_(1), rd=meas_package.raw_measurements_(2);
      double px=r*cos(theta), py=r*sin(theta), vx=rd*cos(theta), vy=rd*sin(theta);
      x_ << px, py, sqrt(vx*vx + vy*vy), 0.0, 0.0;
      P_ << std_radr_*std_radr_,0.0,0.0,0.0,0.0,
            0.0,std_radr_*std_radr_,0.0,0.0,0.0,
            0.0,0.0,std_radrd_*std_radrd_,0.0,0.0,
            0.0,0.0,0.0,1.0,0.0,
            0.0,0.0,0.0,0.0,1.0;
    }
    is_initialized_=true;
  }
  else
  {
    double delta_t=static_cast<double>(meas_package.timestamp_ - time_us_)/1000000.0;
    Prediction(delta_t);
    if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
      UpdateLidar(meas_package);
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    {
      UpdateRadar(meas_package);
    }
  }
  time_us_=meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  ComputeAugmentedSigmaPoints();

  for(int i=0;i<2*n_aug_+1;++i)
  {
    double px=Xsig_aug_(0,i), py=Xsig_aug_(1,i), v=Xsig_aug_(2,i), phi=Xsig_aug_(3,i), phid=Xsig_aug_(4,i), n_a=Xsig_aug_(5,i), n_phidd=Xsig_aug_(6,i);
    
    double px_pred, py_pred;
    if(abs(phid)<0.001)
    {
      px_pred=px + v*cos(phi)*delta_t + 0.5*n_a*cos(phi)*delta_t*delta_t;
      py_pred=py + v*sin(phi)*delta_t + 0.5*n_a*sin(phi)*delta_t*delta_t;
    }
    else
    {
      px_pred=px + v/phid * (sin(phi+phid*delta_t)-sin(phi)) + 0.5*n_a*cos(phi)*delta_t*delta_t;
      py_pred=py + v/phid * (-cos(phi+phid*delta_t)+cos(phi)) + 0.5*n_a*sin(phi)*delta_t*delta_t;
    }
    double v_pred=v + n_a*delta_t;
    double phi_pred=phi + phid*delta_t + 0.5*n_phidd*delta_t*delta_t;
    double phid_pred=phid + n_phidd*delta_t;

    Xsig_pred_(0,i)=px_pred;
    Xsig_pred_(1,i)=py_pred;
    Xsig_pred_(2,i)=v_pred;
    Xsig_pred_(3,i)=phi_pred;
    Xsig_pred_(4,i)=phid_pred;
  }

  x_=Xsig_pred_*weights_;

  XD_=Xsig_pred_.colwise() - x_;
  for(int i=0;i<2*n_aug_+1;++i) NormalizeAngle(XD_(3,i));
  P_=XD_ * wMat_ * XD_.transpose(); 
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z=2;

  MatrixXd H_lidar(n_z,n_x_);
  H_lidar << 1,0,0,0,0,
              0,1,0,0,0;

  MatrixXd Zsig=H_lidar * Xsig_pred_;

  VectorXd z_pred=Zsig * weights_;

  MatrixXd S(n_z,n_z);
  MatrixXd ZD=Zsig.colwise() - z_pred;
  S=(ZD * wMat_ * ZD.transpose()) + R_lidar_;

  MatrixXd Tc=(XD_ * wMat_ * ZD.transpose());
  MatrixXd K=(Tc * S.inverse());

  VectorXd z_diff=(meas_package.raw_measurements_ - z_pred);
  x_ = x_ + K*z_diff;
  P_ = P_ - (K * S * K.transpose());
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  int n_z=3;

  MatrixXd Zsig(n_z,2*n_aug_+1);
  for(int i=0;i<2*n_aug_+1;++i)
  {
    double px=Xsig_pred_(0,i), py=Xsig_pred_(1,i), v=Xsig_pred_(2,i), phi=Xsig_pred_(3,i), phid=Xsig_pred_(4,i);
    double vx=v*cos(phi), vy=v*sin(phi);
    double d=sqrt(px*px + py*py);

    Zsig(0,i)=d;
    Zsig(1,i)=atan2(py,px);
    Zsig(2,i)=(px*vx + py*vy)/d;
  }

  VectorXd z_pred=Zsig * weights_;

  MatrixXd S(n_z,n_z);
  MatrixXd ZD=Zsig.colwise() - z_pred;
  for(int i=0;i<2*n_aug_+1;++i) NormalizeAngle(ZD(1,i));
  S=(ZD * wMat_ * ZD.transpose()) + R_radar_;

  MatrixXd Tc=(XD_ * wMat_ * ZD.transpose());
  MatrixXd K=(Tc * S.inverse());

  VectorXd z_diff=(meas_package.raw_measurements_ - z_pred);
  NormalizeAngle(z_diff(1));
  x_ = x_ + K*z_diff;
  P_ = P_ - (K * S * K.transpose());
}