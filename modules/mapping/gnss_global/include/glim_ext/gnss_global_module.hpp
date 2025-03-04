#include <deque>
#include <atomic>
#include <thread>
#include <numeric>
#include <Eigen/Core>
#include <cmath> 

#define GLIM_ROS2

#include <boost/format.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>

#ifdef GLIM_ROS2
#include <glim/util/extension_module_ros2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS2;
using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
using PoseWithCovarianceStampedConstPtr = geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr;
using NavSatFix = sensor_msgs::msg::NavSatFix;
using NavSatFixConstPtr = sensor_msgs::msg::NavSatFix::ConstSharedPtr;

template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}
#else
#include <glim/util/extension_module_ros.hpp>
#include <geometry_msgs/PoseWithCovarianceStamped.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS;
#endif

#include <spdlog/spdlog.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <glim/util/logging.hpp>
#include <glim/util/convert_to_string.hpp>
#include <glim_ext/util/config_ext.hpp>

namespace glim {

using gtsam::symbol_shorthand::X;

/**
 * @brief Implementation of GNSS constraints for the global optimization.
 * @note  
 *        
 */
class GNSSGlobal : public ExtensionModuleBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GNSSGlobal() : logger(create_module_logger("gnss_global")) {
    logger->info("initializing GNSS global constraints");
    const std::string config_path = glim::GlobalConfigExt::get_config_path("config_gnss_global");
    logger->info("gnss_global_config_path={}", config_path);

    glim::Config config(config_path);
    gnss_topic = config.param<std::string>("gnss", "gnss_topic", "/pose_with_cov");
    prior_inf_scale = config.param<Eigen::Vector3d>("gnss", "prior_inf_scale", Eigen::Vector3d(1e3, 1e3, 0.0));
    min_baseline = config.param<double>("gnss", "min_baseline", 5.0);

    gnss_meas_type = config.param<std::string>("gnss", "gnss_meas_type", "enu");

    T_base_link_gnss = config.param<Eigen::Isometry3d>("gnss", "T_base_link_gnss", Eigen::Isometry3d::Identity());

    transformation_initialized = false;
    T_world_utm.setIdentity();

    kill_switch = false;
    thread = std::thread([this] { backend_task(); });

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    GlobalMappingCallbacks::on_insert_submap.add(std::bind(&GNSSGlobal::on_insert_submap, this, _1));
    GlobalMappingCallbacks::on_smoother_update.add(std::bind(&GNSSGlobal::on_smoother_update, this, _1, _2, _3));
  }
  ~GNSSGlobal() {
    kill_switch = true;
    thread.join();
  }

  virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override {
    // const auto sub = std::make_shared<TopicSubscription<PoseWithCovarianceStamped>>(gnss_topic, [this](const PoseWithCovarianceStampedConstPtr msg) { gnss_callback(msg); });
    if(gnss_meas_type.compare("enu") == 0){
      logger->info("Working on!!! IT'll crash :)");
      const auto sub = std::make_shared<TopicSubscription<NavSatFix>>(gnss_topic, [this](const NavSatFixConstPtr msg) {navsatfix_callback(msg); }); 
      return {sub};
    } else if(gnss_meas_type.compare("lla") == 0) {
      const auto sub = std::make_shared<TopicSubscription<NavSatFix>>(gnss_topic, [this](const NavSatFixConstPtr msg) {navsatfix_callback(msg); }); 
      return {sub};
    } else {
      logger->error("[ERROR] gnss_meas_type not supported! Choose between 'lla' and 'enu'");
      return {};    // Returning an empty vector to handle the error case
    }
  }

  void gnss_callback(const PoseWithCovarianceStampedConstPtr& gnss_msg) {
    Eigen::Vector4d gnss_data;
    const double stamp = to_sec(gnss_msg->header.stamp);
    const auto& pos = gnss_msg->pose.pose.position;
    gnss_data << stamp, pos.x, pos.y, pos.z;
    input_gnss_queue.push_back(gnss_data);
  }

    void navsatfix_callback(const NavSatFixConstPtr& nav_sat_fix_msg) {

    double x, y, z;
    double lat0, lon0, alt0;
    float cov_x, cov_y, cov_z;

    lat0 = 45.519313459310524;        // coordinates of A2A Brescia (taken from database json a2a_completo)
    lon0 = 10.211790103879785;
    alt0 = 170.38059997558594;

    // std::cout << ' nav_sat_fix_msg->latitude {}' << nav_sat_fix_msg->latitude << std::endl;
    convert_lla_2_enu(nav_sat_fix_msg->latitude, nav_sat_fix_msg->longitude, nav_sat_fix_msg->altitude, lat0, lon0, alt0, x, y, z);

    PoseWithCovarianceStamped gnss_msg;

    cov_x = nav_sat_fix_msg->position_covariance[0]; // Covariance for x
    cov_y = nav_sat_fix_msg->position_covariance[4]; // Covariance for y
    cov_z = nav_sat_fix_msg->position_covariance[8]; // Covariance for z
    
    // // bring the gnss data in base link reference frame
    Eigen::Vector3d point(x,y,z);
    Eigen::Vector3d transformed_gnss_data = T_base_link_gnss * point;

    Eigen::Vector4d gnss_data;
    Eigen::Vector4d gnss_covariance;
    const double stamp = to_sec(nav_sat_fix_msg->header.stamp);
    // logger->info("Covariances: {}, {}, {}", cov_x, cov_y, cov_z);
    gnss_data << stamp, transformed_gnss_data.x(), transformed_gnss_data.y(), transformed_gnss_data.z();
    gnss_covariance << stamp, cov_x, cov_y, cov_z;

    // Push GNSS data and covariances into respective queues
    input_gnss_queue.push_back(gnss_data);
    gnss_covariances_queue.push_back(gnss_covariance);
  }

  void convert_lla_2_enu(double lat, double lon, double alt, double lat0, double lon0, double alt0,
		             double &easting, double &northing, double &up) {
    // Initialise sines and cosines
    static const double DEG2RAD = M_PI/180;
    static const double EARTHSEMIMAJOR = 6378137; // Ellipsoids semi-major ax
	  static const double EARTHECCEN2 = 0.00669438; // Ellipsoids eccentricity^2


    double clatRef = cos(lat0  * DEG2RAD);
    double clonRef = cos(lon0  * DEG2RAD);
    double slatRef = sin(lat0   * DEG2RAD);
    double slonRef = sin(lon0  * DEG2RAD);
    double clat    = cos(lat  * DEG2RAD);
    double clon    = cos(lon * DEG2RAD);
    double slat    = sin(lat  * DEG2RAD);
    double slon    = sin(lon * DEG2RAD);
  
    // Compute reference position vector in ECEF coordinates.
    double r0Ref      = EARTHSEMIMAJOR / (sqrt((1.0 - EARTHECCEN2 * slatRef * slatRef)));
    double ecefRef[3];
    double dECEF[3];
    ecefRef[0] = (alt0 + r0Ref) * clatRef * clonRef;             // x-coord
    ecefRef[1] = (alt0 + r0Ref) * clatRef * slonRef;             // y-coord
    ecefRef[2] = (alt0 + r0Ref * (1.0 - EARTHECCEN2)) * slatRef; // z-coord
  
    // Compute data position vectors relative to reference point in ECEF co-ordinates.
    double r0       = EARTHSEMIMAJOR / (sqrt((1.0 - EARTHECCEN2 * slat * slat)));
    dECEF[0] = (alt + r0) * clat * clon - ecefRef[0];                // x-coord
    dECEF[1] = (alt + r0) * clat * slon - ecefRef[1];                // y-coord
    dECEF[2] = (alt + r0 * (1.0 - EARTHECCEN2)) * slat - ecefRef[2]; // z-coord
  
    // Define rotation from ECEF to ENU
    double R[3][3] = {{-slonRef, clonRef, 0}, {-slatRef * clonRef, -slatRef * slonRef, clatRef}, {clatRef * clonRef, clatRef * slonRef, slatRef}};
  
    std::vector<double> enu{0,0,0};
    // Matrix multiplication
    for (int row = 0; row < 3; row++){
      enu[row] = 0.0;
      for (int col = 0; col < 3; col++)
      {
          enu[row] += R[row][col] * dECEF[col];
      }
    }
  
    easting = enu[0];
    northing = enu[1];
    up = enu[2];
  }

  void on_insert_submap(const SubMap::ConstPtr& submap) { 
    input_submap_queue.push_back(submap); 
  }

  void on_smoother_update(gtsam_points::ISAM2Ext& isam2, gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values) {
    const auto factors = output_factors.get_all_and_clear();
    if (!factors.empty()) {
      logger->debug("insert {} GNSS prior factors", factors.size());
      new_factors.add(factors);
    }
  }

  void backend_task() {
    logger->info("starting GNSS global thread");
    std::deque<Eigen::Vector4d> utm_queue;
    std::deque<Eigen::Vector4d> utm_cov_queue;
    std::deque<SubMap::ConstPtr> submap_queue;

    while (!kill_switch) {

      if(gnss_meas_type.compare("lla") == 0){
        // Convert GeoPoint(lat/lon) to UTM
        const auto gnss_data = input_gnss_queue.get_all_and_clear();
        const auto gnss_covariances = gnss_covariances_queue.get_all_and_clear();
        utm_queue.insert(utm_queue.end(), gnss_data.begin(), gnss_data.end());
        utm_cov_queue.insert(utm_cov_queue.end(), gnss_covariances.begin(), gnss_covariances.end());

        // Add new submaps
        const auto new_submaps = input_submap_queue.get_all_and_clear();
        if (new_submaps.empty()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
          continue;
        }
        submap_queue.insert(submap_queue.end(), new_submaps.begin(), new_submaps.end());

        // Remove submaps that are created earlier than the oldest GNSS data
        while (!utm_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp < utm_queue.front()[0]) {
          submap_queue.pop_front();
        }

        // Interpolate UTM coords and associate with submaps
        while (!utm_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp > utm_queue.front()[0] &&
            submap_queue.front()->frames.back()->stamp < utm_queue.back()[0]) {
          const auto& submap = submap_queue.front();
          const double stamp = submap->frames[submap->frames.size() / 2]->stamp;
  
          // find the closest right element
          const auto right = std::lower_bound(utm_queue.begin(), utm_queue.end(), stamp, [](const Eigen::Vector4d& utm, const double t) { return utm[0] < t; });
          if (right == utm_queue.end() || (right + 1) == utm_queue.end()) {
              logger->warn("invalid condition in GNSS global module!!");
              break;
          }
          const auto left = right - 1;

          const auto cov_right = utm_cov_queue.begin() + (right - utm_queue.begin());
          const auto cov_left = cov_right - 1;
              
          logger->debug("submap={:.6f} utm_left={:.6f} utm_right={:.6f}", stamp, (*left)[0], (*right)[0]);

          const double tl = (*left)[0];
          const double tr = (*right)[0];
          const double p = (stamp - tl) / (tr - tl);
          const Eigen::Vector4d interpolated_utm = (1.0 - p) * (*left) + p * (*right);
          const Eigen::Vector4d interpolated_cov = (1.0 - p) * (*cov_left) + p * (*cov_right);

          submaps.push_back(submap);
          submap_coords.push_back(interpolated_utm);
          submap_covariances.push_back(interpolated_cov);

          submap_queue.pop_front();
          utm_queue.erase(utm_queue.begin(), left);
          utm_cov_queue.erase(utm_cov_queue.begin(), cov_left);
        }

        // Initialize T_world_utm if is not already intialized and if enough space has been travelled and the covariance of the gps is small enough
        if (!transformation_initialized && !submaps.empty() && 
            (submaps.front()->T_world_origin.inverse() * submaps.back()->T_world_origin).translation().norm() > min_baseline) { 
              
          const Eigen::Vector3d gnss_cov = submap_covariances.back().tail<3>();

          if (gnss_cov(0) <= 0.001 && gnss_cov(1) <= 0.001 && gnss_cov(2) <= 0.01) {
            Eigen::Vector3d mean_est = Eigen::Vector3d::Zero();
            Eigen::Vector3d mean_gnss = Eigen::Vector3d::Zero();
            for (int i = 0; i < submaps.size(); i++) {
                mean_est += submaps[i]->T_world_origin.translation();
                // mean_gnss += submap_coords[i].tail<3>();
            }
            mean_est /= submaps.size();
            // mean_gnss /= submaps.size();

            // mean_est = submaps.back()->T_world_origin.translation();
            mean_gnss = submap_coords.back().tail<3>();

            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (int i = 0; i < submaps.size(); i++) {
                const Eigen::Vector3d centered_est = submaps[i]->T_world_origin.translation() - mean_est;
                const Eigen::Vector3d centered_gnss = submap_coords[i].tail<3>() - mean_gnss;
                cov += centered_gnss * centered_est.transpose();
            }
            cov /= submaps.size();

            const Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov.block<2, 2>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Eigen::Matrix2d U = svd.matrixU();
            const Eigen::Matrix2d V = svd.matrixV();
            const Eigen::Matrix2d D = svd.singularValues().asDiagonal();
            Eigen::Matrix2d S = Eigen::Matrix2d::Identity();

            const double det = U.determinant() * V.determinant();
            if (det < 0.0) {
                S(1, 1) = -1;
            }

            Eigen::Isometry3d T_utm_world = Eigen::Isometry3d::Identity();
            T_utm_world.linear().block<2, 2>(0, 0) = U * S * V.transpose();
            T_utm_world.translation() = mean_gnss - T_utm_world.linear() * mean_est;

            T_world_utm = T_utm_world.inverse();

            for (int i = 0; i < submaps.size(); i++) {
                const Eigen::Vector3d gnss = T_world_utm * submap_coords[i].tail<3>();
                logger->debug("submap={} gnss={}", convert_to_string(submaps[i]->T_world_origin.translation().eval()), convert_to_string(gnss));
            }

            logger->info("T_world_utm={}", convert_to_string(T_world_utm));
            transformation_initialized = true;
          }
        }

        // Add GPS factor
        if (transformation_initialized) {
            const Eigen::Vector3d xyz = submap_coords.back().tail<3>();
            const Eigen::Vector3d gnss_cov = submap_covariances.back().tail<3>();

            logger->debug("submap={} gnss={}, gnss_cov={}", convert_to_string(submaps.back()->T_world_origin.translation().eval()), convert_to_string(xyz), convert_to_string(gnss_cov));

            const auto& submap = submaps.back();           

            // insert the gnss pose in the graph only if the covariance is smaller than a treshold
            if (gnss_cov(0) < 1.0 && gnss_cov(1) < 1.0 && gnss_cov(2) < 1.0) {  

              gtsam::Vector Vector3(3);
              // Vector3 << max(gnss_cov(0), 1.0f), max( gnss_cov(1), 1.0f), max( gnss_cov(2), 1.0f);
              Vector3 << gnss_cov(0), gnss_cov(1), gnss_cov(2);

              gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);

              logger->info("Adding GNSS pose with noise: {}", convert_to_string(Vector3));

              gtsam::NonlinearFactor::shared_ptr factor(new gtsam::GPSFactor(X(submap->id), gtsam::Point3(xyz), gps_noise));

              output_factors.push_back(factor);
            }
         }
      } else if(gnss_meas_type.compare("enu") == 0){ 
        logger->info("WORKING ONN!!!!");
      } else {
        logger->error("[ERROR] gnss_meas_type not supported! Choose between 'lla' and 'enu'");
      }

    }
  }

private:
  std::atomic_bool kill_switch;
  std::thread thread;

  ConcurrentVector<Eigen::Vector4d> input_gnss_queue;
  ConcurrentVector<SubMap::ConstPtr> input_submap_queue;
  ConcurrentVector<Eigen::Vector4d> gnss_covariances_queue;
  ConcurrentVector<gtsam::NonlinearFactor::shared_ptr> output_factors;

  std::vector<SubMap::ConstPtr> submaps;
  std::vector<Eigen::Vector4d> submap_coords;
  std::vector<Eigen::Vector4d> submap_covariances;

  std::string gnss_topic;
  std::string gnss_meas_type;
  Eigen::Isometry3d T_base_link_gnss;
  Eigen::Vector3d prior_inf_scale;
  double min_baseline;

  bool transformation_initialized;
  Eigen::Isometry3d T_world_utm;

  // Logging
  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::GNSSGlobal();
}