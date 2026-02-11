# Chapter 4: Navigation and Planning with Isaac and Nav2

## Overview

In this chapter, we'll explore how to implement navigation and planning for humanoid robots using Isaac and Nav2 (Navigation Stack 2). We'll cover autonomous navigation, path planning, and obstacle avoidance in complex environments.

## Introduction to Navigation in Humanoid Robotics

Navigation for humanoid robots presents unique challenges:

- **Dynamic balance**: Maintaining stability while moving
- **Complex kinematics**: Multi-degree-of-freedom leg systems
- **Terrain adaptation**: Navigating uneven surfaces
- **Social navigation**: Moving safely around humans
- **Energy efficiency**: Optimizing gait for battery life

## Nav2 Architecture Overview

Nav2 is the navigation stack for ROS 2, providing:

- **Global Planner**: Computes optimal path from start to goal
- **Local Planner**: Executes path while avoiding obstacles
- **Controller**: Translates plans to robot commands
- **Behavior Trees**: Coordinates navigation behaviors
- **Map Management**: Handles occupancy grids and semantic maps

### Nav2 Components

```yaml
# Example Nav2 launch configuration
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    likelihood_max_dist: 2.0
    set_initial_pose: true
    initial_pose:
      x: 0.0
      y: 0.0
      z: 0.0
      yaw: 0.0
    sigma_hit: 0.2

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "nav2_bt_xml_v04/navigate_through_poses_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "nav2_bt_xml_v04/navigate_to_pose_w_replanning_and_recovery.xml"
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_smooth_path_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_assisted_teleop_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_drive_on_heading_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_globally_updated_goal_condition_bt_node
      - nav2_is_path_valid_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_truncate_path_local_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_distance_traveled_condition_bt_node
      - nav2_single_trigger_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node
      - nav2_controller_cancel_bt_node
      - nav2_path_longer_on_approach_bt_node
      - nav2_wait_cancel_bt_node
      - nav2_spin_cancel_bt_node
      - nav2_back_up_cancel_bt_node
      - nav2_assisted_teleop_cancel_bt_node
      - nav2_drive_on_heading_cancel_bt_node
```

## Isaac Integration with Nav2

### Isaac Navigation Components

Isaac provides specialized navigation components for humanoid robots:

```python
# Isaac Nav2 integration example
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, PointCloud2
from std_srvs.srv import Empty
import numpy as np

class IsaacHumanoidNavigator(Node):
    def __init__(self):
        super().__init__('isaac_humanoid_navigator')
        
        # Navigation state
        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.path = []
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'local_plan', 10)
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, 'amcl_pose', self.pose_callback, 10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        
        # Services
        self.nav_to_pose_cli = self.create_client(
            interfaces.srv.NavigateToPose, 'navigate_to_pose'
        )
        
        # Timers
        self.nav_timer = self.create_timer(0.1, self.navigation_callback)
        
        # Isaac-specific navigation parameters
        self.setup_humanoid_navigation_params()
    
    def setup_humanoid_navigation_params(self):
        """Configure navigation parameters specific to humanoid robots"""
        # Humanoid-specific parameters
        self.max_linear_speed = 0.5  # Slower for stability
        self.max_angular_speed = 0.5
        self.min_turn_radius = 0.3   # Limited by leg kinematics
        self.step_height = 0.1       # Maximum step height capability
        
        # Balance constraints
        self.balance_margin = 0.1    # Safety margin for COM
        self.zmp_tolerance = 0.05    # Zero Moment Point tolerance
    
    def pose_callback(self, msg):
        self.current_pose = msg.pose
    
    def laser_callback(self, msg):
        # Process laser data for obstacle detection
        self.detect_obstacles_laser(msg)
    
    def map_callback(self, msg):
        self.map = msg
    
    def detect_obstacles_laser(self, laser_msg):
        """Detect obstacles using laser scanner"""
        # Convert laser ranges to Cartesian coordinates
        angles = np.linspace(
            laser_msg.angle_min, 
            laser_msg.angle_max, 
            len(laser_msg.ranges)
        )
        
        # Filter valid ranges
        valid_ranges = [(angle, dist) for angle, dist in zip(angles, laser_msg.ranges) 
                        if laser_msg.range_min <= dist <= laser_msg.range_max]
        
        # Convert to obstacle points
        obstacle_points = []
        for angle, dist in valid_ranges:
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            obstacle_points.append((x, y))
        
        # Check if obstacles are within humanoid's reach
        for x, y in obstacle_points:
            distance = np.sqrt(x**2 + y**2)
            if distance < self.step_height * 2:  # Potential step obstacle
                self.handle_step_obstacle(x, y, distance)
    
    def handle_step_obstacle(self, x, y, distance):
        """Handle obstacles that require stepping over"""
        # Determine if the obstacle can be stepped over
        if distance < self.step_height:
            # Plan step-over maneuver
            self.plan_step_maneuver(x, y)
    
    def plan_step_maneuver(self, obstacle_x, obstacle_y):
        """Plan a maneuver to step over an obstacle"""
        # This would involve complex humanoid kinematics
        # For now, we'll create a simple detour
        detour_offset = 0.3  # meters
        
        # Calculate detour waypoints
        detour_waypoints = [
            (obstacle_x - detour_offset, obstacle_y),
            (obstacle_x, obstacle_y + detour_offset),
            (obstacle_x + detour_offset, obstacle_y)
        ]
        
        # Publish detour path
        self.publish_detour_path(detour_waypoints)
    
    def navigation_callback(self):
        """Main navigation callback"""
        if self.current_pose and self.goal_pose:
            # Check if we need to replan
            if self.should_replan():
                self.compute_new_path()
            
            # Follow the path
            self.follow_path()
    
    def should_replan(self):
        """Determine if path replanning is needed"""
        # Check for new obstacles
        # Check for significant deviation from path
        # Check for goal updates
        return False  # Simplified for example
    
    def compute_new_path(self):
        """Compute a new path to the goal"""
        # This would call Nav2's global planner
        # For humanoid robots, we might need specialized planners
        pass
    
    def follow_path(self):
        """Follow the current path"""
        # This would call Nav2's local planner
        # But adapted for humanoid kinematics
        pass
```

## Humanoid-Specific Navigation Challenges

### Balance-Aware Path Planning

Humanoid robots must consider balance during navigation:

```cpp
// C++ example for balance-aware navigation
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>

class BalanceAwarePlanner {
public:
    BalanceAwarePlanner() {
        // Initialize humanoid-specific parameters
        max_step_length_ = 0.3;      // Maximum step length
        max_step_width_ = 0.2;       // Maximum step width
        com_height_ = 0.8;           // Center of mass height
        foot_separation_ = 0.2;      // Distance between feet
    }
    
    nav_msgs::msg::Path planBalancedPath(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal,
        const nav_msgs::msg::OccupancyGrid& map) {
        
        nav_msgs::msg::Path balanced_path;
        
        // Generate candidate path using standard algorithm
        nav_msgs::msg::Path raw_path = generateRawPath(start, goal, map);
        
        // Modify path for balance constraints
        for (size_t i = 0; i < raw_path.poses.size(); ++i) {
            geometry_msgs::msg::PoseStamped balanced_pose;
            
            // Check if pose violates balance constraints
            if (isValidStep(raw_path.poses[i], i > 0 ? raw_path.poses[i-1] : start)) {
                balanced_pose = raw_path.poses[i];
            } else {
                // Find alternative balanced pose
                balanced_pose = findBalancedAlternative(raw_path.poses[i], map);
            }
            
            balanced_path.poses.push_back(balanced_pose);
        }
        
        return balanced_path;
    }

private:
    bool isValidStep(const geometry_msgs::msg::PoseStamped& current_pose,
                    const geometry_msgs::msg::PoseStamped& previous_pose) {
        // Calculate step dimensions
        double dx = current_pose.pose.position.x - previous_pose.pose.position.x;
        double dy = current_pose.pose.position.y - previous_pose.pose.position.y;
        double step_distance = sqrt(dx*dx + dy*dy);
        
        // Check step length constraint
        if (step_distance > max_step_length_) {
            return false;
        }
        
        // Check if step is within support polygon
        // This is a simplified check
        return true;
    }
    
    geometry_msgs::msg::PoseStamped findBalancedAlternative(
        const geometry_msgs::msg::PoseStamped& original_pose,
        const nav_msgs::msg::OccupancyGrid& map) {
        
        // Search for nearby poses that satisfy balance constraints
        geometry_msgs::msg::PoseStamped best_pose = original_pose;
        
        // Grid search around original pose
        for (double offset_x = -0.1; offset_x <= 0.1; offset_x += 0.05) {
            for (double offset_y = -0.1; offset_y <= 0.1; offset_y += 0.05) {
                geometry_msgs::msg::PoseStamped candidate_pose = original_pose;
                candidate_pose.pose.position.x += offset_x;
                candidate_pose.pose.position.y += offset_y;
                
                // Check if candidate is valid and improves balance
                if (isPoseValid(candidate_pose, map) && 
                    isBalancedStep(candidate_pose, original_pose)) {
                    
                    // Calculate score based on path optimality and balance
                    double score = calculateBalanceScore(candidate_pose);
                    double current_score = calculateBalanceScore(best_pose);
                    
                    if (score > current_score) {
                        best_pose = candidate_pose;
                    }
                }
            }
        }
        
        return best_pose;
    }
    
    bool isBalancedStep(const geometry_msgs::msg::PoseStamped& current,
                        const geometry_msgs::msg::PoseStamped& previous) {
        // Implement balance constraint checking
        // This would involve ZMP (Zero Moment Point) calculations
        return true; // Simplified for example
    }
    
    nav_msgs::msg::Path generateRawPath(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal,
        const nav_msgs::msg::OccupancyGrid& map) {
        // Implementation of standard path planning algorithm
        // (e.g., A*, Dijkstra, etc.)
        nav_msgs::msg::Path path;
        // Add poses to path
        return path;
    }
    
    bool isPoseValid(const geometry_msgs::msg::PoseStamped& pose,
                     const nav_msgs::msg::OccupancyGrid& map) {
        // Check if pose is collision-free
        int map_x = (pose.pose.position.x - map.info.origin.position.x) / map.info.resolution;
        int map_y = (pose.pose.position.y - map.info.origin.position.y) / map.info.resolution;
        
        int index = map_y * map.info.width + map_x;
        if (index >= 0 && index < static_cast<int>(map.data.size())) {
            return map.data[index] < 50; // Free space threshold
        }
        return false;
    }
    
    double calculateBalanceScore(const geometry_msgs::msg::PoseStamped& pose) {
        // Calculate score based on balance metrics
        return 1.0; // Simplified for example
    }
    
    double max_step_length_;
    double max_step_width_;
    double com_height_;
    double foot_separation_;
};
```

### Terrain Classification and Adaptation

Humanoid robots need to adapt to different terrains:

```python
# Terrain classification for humanoid navigation
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class TerrainClassifier:
    def __init__(self):
        # Initialize terrain classifier
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.terrain_types = {
            0: 'flat_ground',
            1: 'uneven_ground', 
            2: 'stairs',
            3: 'slope',
            4: 'obstacle'
        }
        
        # Training data placeholders (would be loaded from real data)
        self.training_features = []
        self.training_labels = []
    
    def classify_terrain(self, point_cloud_msg):
        """Classify terrain based on point cloud data"""
        # Extract features from point cloud
        points = list(pc2.read_points(point_cloud_msg, 
                                    field_names=("x", "y", "z"), 
                                    skip_nans=True))
        
        if len(points) < 10:  # Not enough points for classification
            return 'unknown'
        
        # Convert to numpy array
        points_array = np.array(points)
        
        # Extract terrain features
        features = self.extract_terrain_features(points_array)
        
        # Classify terrain
        prediction = self.classifier.predict([features])[0]
        return self.terrain_types[prediction]
    
    def extract_terrain_features(self, points):
        """Extract features for terrain classification"""
        # Height variance
        height_variance = np.var(points[:, 2])
        
        # Roughness measure
        z_diffs = np.diff(np.sort(points[:, 2]))
        roughness = np.mean(z_diffs) if len(z_diffs) > 0 else 0
        
        # Planarity (fit a plane and measure residuals)
        plane_fit_error = self.fit_plane_measure_error(points)
        
        # Density of points
        density = len(points) / self.calculate_convex_hull_volume(points)
        
        return [height_variance, roughness, plane_fit_error, density]
    
    def fit_plane_measure_error(self, points):
        """Fit a plane to points and return mean squared error"""
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Center points
        centered_points = points - centroid
        
        # Singular value decomposition
        _, _, vh = np.linalg.svd(centered_points)
        
        # Normal vector of best-fit plane
        normal = vh[2, :]
        
        # Calculate distances to plane
        distances = np.abs(np.dot(points - centroid, normal))
        
        return np.mean(distances**2)
    
    def calculate_convex_hull_volume(self, points):
        """Calculate approximate volume using bounding box"""
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        volume = np.prod(maxs - mins)
        return max(volume, 1.0)  # Return at least 1.0 to avoid division by zero

class AdaptiveGaitController:
    def __init__(self):
        self.terrain_classifier = TerrainClassifier()
        self.current_terrain = 'flat_ground'
        
        # Gait parameters for different terrains
        self.gait_params = {
            'flat_ground': {
                'step_length': 0.3,
                'step_height': 0.05,
                'step_time': 0.8,
                'balance_margin': 0.1
            },
            'uneven_ground': {
                'step_length': 0.2,
                'step_height': 0.1,
                'step_time': 1.0,
                'balance_margin': 0.15
            },
            'stairs': {
                'step_length': 0.25,
                'step_height': 0.15,
                'step_time': 1.2,
                'balance_margin': 0.2
            },
            'slope': {
                'step_length': 0.25,
                'step_height': 0.08,
                'step_time': 0.9,
                'balance_margin': 0.12
            },
            'obstacle': {
                'step_length': 0.15,
                'step_height': 0.2,
                'step_time': 1.5,
                'balance_margin': 0.25
            }
        }
    
    def update_terrain(self, point_cloud_msg):
        """Update current terrain classification"""
        new_terrain = self.terrain_classifier.classify_terrain(point_cloud_msg)
        if new_terrain != self.current_terrain:
            self.current_terrain = new_terrain
            self.adjust_gait_for_terrain(new_terrain)
    
    def adjust_gait_for_terrain(self, terrain_type):
        """Adjust gait parameters based on terrain"""
        params = self.gait_params.get(terrain_type, self.gait_params['flat_ground'])
        
        # Publish gait parameters to humanoid controller
        self.publish_gait_parameters(params)
    
    def publish_gait_parameters(self, params):
        """Publish gait parameters to humanoid controller"""
        # This would publish to appropriate ROS topics
        pass
```

## Isaac Sim Navigation Testing

### Creating Navigation Scenarios

Test navigation in Isaac Sim with various scenarios:

```python
# Isaac Sim navigation scenario
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.range_sensor import attach_lidar_sensor
import numpy as np

class IsaacNavigationScenario:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_scenario()
    
    def setup_scenario(self):
        """Set up navigation scenario in Isaac Sim"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add robot
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets path")
            return
        
        # Add a simple robot (in practice, use a humanoid model)
        robot_asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        add_reference_to_stage(
            usd_path=robot_asset_path,
            prim_path="/World/Robot"
        )
        
        # Add obstacles
        self.add_navigation_obstacles()
        
        # Add lidar sensor
        attach_lidar_sensor(
            prim_path="/World/Robot/base_link",
            sensor_tick_period=0.1
        )
        
        # Play the simulation
        self.world.reset()
    
    def add_navigation_obstacles(self):
        """Add obstacles for navigation testing"""
        # Add various obstacles
        obstacle_configs = [
            {"name": "box1", "position": [2.0, 0.0, 0.5], "size": [0.5, 0.5, 1.0]},
            {"name": "box2", "position": [3.0, 1.0, 0.3], "size": [0.3, 0.3, 0.6]},
            {"name": "cylinder1", "position": [1.5, -1.0, 0.5], "radius": 0.3, "height": 1.0}
        ]
        
        for config in obstacle_configs:
            if "size" in config:
                # Add box obstacle
                create_prim(
                    prim_path=f"/World/Obstacles/{config['name']}",
                    prim_type="Cube",
                    position=config["position"],
                    scale=config["size"]
                )
            elif "radius" in config:
                # Add cylinder obstacle
                create_prim(
                    prim_path=f"/World/Obstacles/{config['name']}",
                    prim_type="Cylinder", 
                    position=config["position"],
                    scale=[config["radius"], config["radius"], config["height"]]
                )
    
    def run_navigation_test(self):
        """Run navigation test in Isaac Sim"""
        # Set goal position
        goal_position = [5.0, 0.0, 0.0]
        
        # Initialize navigation system
        # This would connect to ROS Nav2 stack
        print(f"Running navigation test to goal: {goal_position}")
        
        # Run simulation
        for i in range(1000):  # Run for 1000 steps
            self.world.step(render=True)
            
            # Check if robot reached goal
            robot_position = self.get_robot_position()
            distance_to_goal = np.linalg.norm(
                np.array(robot_position[:2]) - np.array(goal_position[:2])
            )
            
            if distance_to_goal < 0.5:  # Within 0.5m of goal
                print(f"Goal reached at step {i}")
                break
    
    def get_robot_position(self):
        """Get current robot position"""
        # This would query the robot's position from Isaac Sim
        # Implementation depends on robot type
        pass

# Example usage
def main():
    scenario = IsaacNavigationScenario()
    scenario.run_navigation_test()

if __name__ == "__main__":
    main()
```

## Behavior Trees for Navigation

### Custom Navigation Behaviors

Create custom behavior trees for humanoid navigation:

```xml
<!-- Custom behavior tree for humanoid navigation -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <Sequence name="NavigateWithHumanoidConstraints">
            <Fallback name="CheckNavigationFeasibility">
                <Condition ID="IsGoalAccessible" />
                <ReactiveSequence name="HandleInaccessibleGoal">
                    <Action ID="ReportNavigationFailure" />
                    <Action ID="RequestHumanAssistance" />
                </ReactiveSequence>
            </Fallback>
            
            <Sequence name="PlanAndExecuteNavigation">
                <Action ID="ComputeGlobalPath" />
                
                <ReactiveFallback name="PathExecution">
                    <Sequence name="NormalPathFollowing">
                        <Action ID="InitializeLocalPlanner" />
                        <ReactiveSequence name="FollowPath">
                            <Action ID="GetNextWaypoint" />
                            <Action ID="CheckBalanceConstraints" />
                            <Action ID="ExecuteStep" />
                            <Condition ID="IsPathComplete" />
                        </ReactiveSequence>
                    </Sequence>
                    
                    <Sequence name="RecoveryBehaviors">
                        <Action ID="ClearLocalCostmap" />
                        <Action ID="SpinInPlace" angle="1.57" />
                        <Action ID="BackUp" distance="0.3" />
                        <Action ID="ComputeGlobalPath" />  <!-- Replan -->
                    </Sequence>
                </ReactiveFallback>
            </Sequence>
        </Sequence>
    </BehaviorTree>
    
    <TreeNodesModel>
        <Action ID="IsGoalAccessible" />
        <Action ID="ReportNavigationFailure" />
        <Action ID="RequestHumanAssistance" />
        <Action ID="ComputeGlobalPath" />
        <Action ID="InitializeLocalPlanner" />
        <Action ID="GetNextWaypoint" />
        <Action ID="CheckBalanceConstraints" />
        <Action ID="ExecuteStep" />
        <Action ID="IsPathComplete" />
        <Action ID="ClearLocalCostmap" />
        <Action ID="SpinInPlace" />
        <Action ID="BackUp" />
    </TreeNodesModel>
</root>
```

## Performance Optimization

### Navigation Performance Metrics

Monitor navigation performance for humanoid robots:

```python
# Navigation performance monitoring
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32
import time
import numpy as np

class NavigationPerformanceMonitor(Node):
    def __init__(self):
        super().__init__('navigation_performance_monitor')
        
        self.start_time = None
        self.path_start_pos = None
        self.current_pos = None
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, 'amcl_pose', self.pose_callback, 10
        )
        self.path_sub = self.create_subscription(
            Path, 'global_plan', self.path_callback, 10
        )
        
        # Publishers for metrics
        self.exec_time_pub = self.create_publisher(Float32, 'navigation_execution_time', 10)
        self.success_rate_pub = self.create_publisher(Float32, 'navigation_success_rate', 10)
        self.energy_efficiency_pub = self.create_publisher(Float32, 'energy_efficiency', 10)
        
        # Navigation statistics
        self.total_attempts = 0
        self.successful_navigations = 0
        self.navigation_times = []
    
    def pose_callback(self, msg):
        self.current_pos = (msg.pose.position.x, msg.pose.position.y)
        
        if self.start_time and self.path_start_pos and self.current_pos:
            # Calculate metrics
            self.calculate_execution_time()
            self.calculate_energy_efficiency()
    
    def path_callback(self, msg):
        if msg.poses:
            self.path_start_pos = (
                msg.poses[0].pose.position.x,
                msg.poses[0].pose.position.y
            )
            self.start_time = time.time()
            self.total_attempts += 1
    
    def calculate_execution_time(self):
        if self.start_time:
            exec_time = time.time() - self.start_time
            time_msg = Float32()
            time_msg.data = exec_time
            self.exec_time_pub.publish(time_msg)
    
    def calculate_energy_efficiency(self):
        # Simplified energy calculation based on movement
        if self.path_start_pos and self.current_pos:
            distance_traveled = np.sqrt(
                (self.current_pos[0] - self.path_start_pos[0])**2 +
                (self.current_pos[1] - self.path_start_pos[1])**2
            )
            
            # Energy efficiency metric (simplified)
            efficiency = distance_traveled / (time.time() - self.start_time) if self.start_time else 0
            
            eff_msg = Float32()
            eff_msg.data = efficiency
            self.energy_efficiency_pub.publish(eff_msg)
    
    def report_success(self):
        self.successful_navigations += 1
        success_rate = self.successful_navigations / self.total_attempts if self.total_attempts > 0 else 0
        
        rate_msg = Float32()
        rate_msg.data = success_rate
        self.success_rate_pub.publish(rate_msg)
```

## Best Practices

### Navigation Design Principles
1. **Safety First**: Always prioritize robot and human safety
2. **Balance Awareness**: Consider humanoid balance constraints in planning
3. **Adaptive Behavior**: Adjust navigation based on terrain and environment
4. **Robust Recovery**: Implement comprehensive recovery behaviors
5. **Performance Monitoring**: Continuously monitor navigation performance

### Implementation Guidelines
1. **Modular Design**: Separate path planning, control, and recovery
2. **Parameter Tuning**: Carefully tune parameters for humanoid kinematics
3. **Simulation Testing**: Extensively test in simulation before real deployment
4. **Gradual Complexity**: Start with simple scenarios and increase complexity
5. **Continuous Learning**: Update navigation strategies based on experience

## Troubleshooting Common Issues

### Navigation Failures
- **Localization errors**: Ensure proper AMCL configuration
- **Path planning failures**: Check map quality and inflation settings
- **Oscillation**: Adjust controller parameters and tolerances
- **Step failures**: Implement proper humanoid-specific gait control

### Performance Issues
- **Slow navigation**: Optimize path planning and control frequency
- **High CPU/GPU usage**: Profile and optimize computational bottlenecks
- **Memory leaks**: Monitor resource usage during long operations

## Summary

In this chapter, we've explored navigation and planning for humanoid robots using Isaac and Nav2. We've covered balance-aware path planning, terrain adaptation, and simulation testing. Navigation for humanoid robots requires special consideration of balance constraints, kinematic limitations, and adaptive gait control. In the next chapter, we'll examine sim-to-real transfer techniques.