# Chapter 3: Humanoid Manipulation and Grasping Strategies

## Overview

In this chapter, we'll explore advanced manipulation and grasping strategies specifically designed for humanoid robots. We'll cover the unique challenges of humanoid manipulation, grasp planning, dual-arm coordination, and adaptive grasping techniques.

## Introduction to Humanoid Manipulation

Humanoid manipulation presents unique challenges compared to traditional robotic manipulators:

- **Anthropomorphic Design**: Human-like arms and hands for human-compatible tasks
- **Dual-Arm Coordination**: Two arms working together for complex tasks
- **Whole-Body Integration**: Coordination between arms, torso, and legs
- **Adaptive Grasping**: Handling objects of various shapes, sizes, and materials
- **Social Compliance**: Performing manipulation in human environments

### Humanoid Manipulation Challenges

```
┌─────────────────────────────────────────────────────────────┐
│                    Humanoid Manipulation                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ Left Arm    │    │ Torso       │    │ Right Arm   │   │
│  │ • Shoulder  │    │ • Balance   │    │ • Shoulder  │   │
│  │ • Elbow     │    │ • Posture   │    │ • Elbow     │   │
│  │ • Wrist     │    │ • Stability │    │ • Wrist     │   │
│  │ • Hand      │    │             │    │ • Hand      │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Dual-Arm Coordination                               │  │
│  │ • Bimanual tasks (opening jars, folding clothes)  │  │
│  │ • Handover operations                             │  │
│  │ • Cooperative manipulation                        │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Anthropomorphic Hand Design

### Human-Inspired Hand Architecture

Humanoid robots often feature anthropomorphic hands designed to mimic human capabilities:

```python
# Anthropomorphic hand model
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Finger:
    name: str
    joints: List[float]  # Joint angles
    links: List[float]   # Link lengths
    joint_limits: List[Tuple[float, float]]  # Min/max joint angles
    dof: int  # Degrees of freedom

class AnthropomorphicHand:
    def __init__(self, hand_type="right"):
        self.hand_type = hand_type
        self.fingers = self.create_finger_structure()
        self.palm = self.define_palm_geometry()
        self.actuators = self.initialize_actuators()
        
    def create_finger_structure(self):
        """Create finger structure mimicking human hand"""
        fingers = {}
        
        # Thumb (opposable, 2 DOF at CMC, 2 at MCP, 1 at IP)
        fingers['thumb'] = Finger(
            name='thumb',
            joints=[0.0, 0.0, 0.0, 0.0],  # 4 joints
            links=[0.03, 0.02, 0.02],     # Link lengths (m)
            joint_limits=[
                (-0.5, 0.5),   # CMC abduction/adduction
                (-0.5, 1.0),   # CMC flexion/extension
                (-0.2, 1.5),   # MCP flexion
                (-0.1, 1.5)    # IP flexion
            ],
            dof=4
        )
        
        # Index, middle, ring, pinky fingers (similar structure)
        for finger_name in ['index', 'middle', 'ring', 'pinky']:
            fingers[finger_name] = Finger(
                name=finger_name,
                joints=[0.0, 0.0, 0.0],  # MCP, PIP, DIP joints
                links=[0.03, 0.02, 0.015],  # Link lengths (m)
                joint_limits=[
                    (-0.2, 1.5),   # MCP flexion
                    (-0.1, 1.5),   # PIP flexion
                    (-0.1, 1.5)    # DIP flexion
                ],
                dof=3
            )
        
        return fingers
    
    def define_palm_geometry(self):
        """Define palm geometry and constraints"""
        return {
            'length': 0.08,    # Palm length (m)
            'width': 0.07,     # Palm width (m)
            'thickness': 0.02, # Palm thickness (m)
            'origin_offset': np.array([0.0, 0.0, 0.0])  # Offset from wrist
        }
    
    def initialize_actuators(self):
        """Initialize hand actuators"""
        actuators = {}
        
        for finger_name, finger in self.fingers.items():
            for i in range(finger.dof):
                actuators[f"{finger_name}_joint_{i}"] = {
                    'type': 'servo',
                    'max_torque': 1.0,  # Nm
                    'max_velocity': 2.0,  # rad/s
                    'gear_ratio': 100,
                    'encoder_resolution': 4096
                }
        
        return actuators
    
    def calculate_finger_kinematics(self, finger_name, joint_angles):
        """Calculate forward kinematics for a finger"""
        finger = self.fingers[finger_name]
        
        if len(joint_angles) != finger.dof:
            raise ValueError(f"Expected {finger.dof} joint angles for {finger_name}")
        
        # Simple kinematic chain calculation
        positions = []
        current_pos = np.array([0.0, 0.0, 0.0])  # Start at finger base
        
        for i, angle in enumerate(joint_angles):
            # Calculate position of this joint/link
            link_length = finger.links[min(i, len(finger.links)-1)]
            
            # For simplicity, assume each joint rotates around z-axis
            # In reality, finger joints have complex axes
            x_offset = link_length * np.cos(angle)
            y_offset = link_length * np.sin(angle)
            
            current_pos += np.array([x_offset, y_offset, 0.0])
            positions.append(current_pos.copy())
        
        return positions

class HandController:
    def __init__(self, hand_model):
        self.hand = hand_model
        self.inverse_kinematics = self.initialize_ik_solver()
    
    def grasp_object(self, object_info, grasp_type="precision"):
        """Plan and execute grasp based on object properties"""
        # Determine optimal grasp configuration
        grasp_config = self.plan_grasp(object_info, grasp_type)
        
        # Execute grasp
        self.execute_grasp(grasp_config)
    
    def plan_grasp(self, object_info, grasp_type):
        """Plan grasp configuration for object"""
        if grasp_type == "precision":
            return self.plan_precision_grasp(object_info)
        elif grasp_type == "power":
            return self.plan_power_grasp(object_info)
        elif grasp_type == "cylindrical":
            return self.plan_cylindrical_grasp(object_info)
        else:
            return self.plan_adaptive_grasp(object_info)
    
    def plan_precision_grasp(self, object_info):
        """Plan precision grasp (thumb-index finger pinch)"""
        # Calculate contact points
        contact_points = self.calculate_contact_points(object_info, "precision")
        
        # Determine joint angles for contact points
        joint_angles = self.inverse_kinematics.solve(contact_points)
        
        return {
            'type': 'precision',
            'contact_points': contact_points,
            'joint_angles': joint_angles,
            'expected_force_distribution': self.calculate_force_distribution(contact_points)
        }
    
    def calculate_contact_points(self, object_info, grasp_type):
        """Calculate optimal contact points for grasp"""
        # This would involve complex geometric calculations
        # For now, return placeholder values
        if grasp_type == "precision":
            return [
                np.array([0.05, 0.02, 0.0]),  # Thumb contact
                np.array([0.05, -0.02, 0.0])  # Index finger contact
            ]
        else:
            return []
```

## Grasp Planning Algorithms

### Geometric Grasp Planning

Geometric approaches to grasp planning analyze object geometry to determine stable grasp points:

```python
# Geometric grasp planner
import open3d as o3d
from scipy.spatial import ConvexHull
import trimesh

class GeometricGraspPlanner:
    def __init__(self):
        self.contact_finder = ContactPointFinder()
        self.stability_evaluator = StabilityEvaluator()
        self.quality_metrics = GraspQualityMetrics()
    
    def plan_grasps(self, object_mesh, max_grasps=10):
        """Plan multiple grasp candidates for object"""
        # Extract object properties
        object_properties = self.analyze_object(object_mesh)
        
        # Find potential contact points
        contact_candidates = self.contact_finder.find_contacts(object_mesh)
        
        # Evaluate grasp stability
        stable_grasps = []
        for contact_pair in self.generate_contact_pairs(contact_candidates):
            grasp_candidate = self.evaluate_grasp_stability(
                contact_pair, object_properties
            )
            
            if grasp_candidate['stable']:
                stable_grasps.append(grasp_candidate)
        
        # Rank grasps by quality
        ranked_grasps = self.rank_grasps(stable_grasps, object_properties)
        
        return ranked_grasps[:max_grasps]
    
    def analyze_object(self, mesh):
        """Analyze object geometric properties"""
        properties = {}
        
        # Calculate bounding box
        bbox = mesh.get_axis_aligned_bounding_box()
        properties['bbox'] = np.asarray(bbox.get_extent())
        properties['center'] = np.asarray(bbox.get_center())
        
        # Calculate volume and mass (assuming uniform density)
        volume = mesh.get_volume()
        properties['volume'] = volume
        properties['mass'] = volume * 1000  # Assuming 1000 kg/m³ density
        
        # Calculate principal axes
        vertices = np.asarray(mesh.vertices)
        cov_matrix = np.cov(vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        properties['principal_axes'] = eigenvectors
        properties['dimensions'] = np.sqrt(eigenvalues)
        
        return properties
    
    def generate_contact_pairs(self, contact_points):
        """Generate pairs of contact points for two-finger grasps"""
        pairs = []
        n_points = len(contact_points)
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                pairs.append((contact_points[i], contact_points[j]))
        
        return pairs
    
    def evaluate_grasp_stability(self, contact_pair, object_properties):
        """Evaluate stability of grasp using contact points"""
        contact1, contact2 = contact_pair
        
        # Calculate grasp axis
        grasp_axis = contact2['position'] - contact1['position']
        grasp_width = np.linalg.norm(grasp_axis)
        
        # Check if grasp width is within hand limits
        max_grasp_width = 0.1  # 10 cm max for adult hand
        if grasp_width > max_grasp_width:
            return {'stable': False, 'reason': 'grasp_too_wide'}
        
        # Calculate grasp center
        grasp_center = (contact1['position'] + contact2['position']) / 2
        
        # Check if grasp center is near object center
        object_center = object_properties['center']
        center_offset = np.linalg.norm(grasp_center - object_center)
        
        # Calculate grasp quality metrics
        quality = self.quality_metrics.evaluate_grasp(
            contact_pair, object_properties
        )
        
        return {
            'stable': quality['antipodal'] and quality['force_closure'],
            'contact_points': contact_pair,
            'grasp_axis': grasp_axis,
            'grasp_width': grasp_width,
            'quality': quality,
            'center_offset': center_offset
        }
    
    def rank_grasps(self, grasps, object_properties):
        """Rank grasps by quality and suitability"""
        def grasp_score(grasp):
            quality = grasp['quality']
            
            # Weight different quality metrics
            score = (
                0.4 * quality['force_closure_quality'] +
                0.3 * quality['volume_displacement'] +
                0.2 * (1.0 - grasp['center_offset'] / np.linalg.norm(object_properties['dimensions'])) +
                0.1 * quality['contact_surface_alignment']
            )
            
            return score
        
        return sorted(grasps, key=grasp_score, reverse=True)

class ContactPointFinder:
    """Find potential contact points on object surface"""
    def find_contacts(self, mesh, density=100):
        """Find contact points on object surface"""
        # Sample points on mesh surface
        surface_points = self.sample_surface_points(mesh, density)
        
        # Filter points based on geometric properties
        contact_points = []
        for point in surface_points:
            properties = self.analyze_surface_properties(mesh, point)
            
            if self.is_good_contact_point(properties):
                contact_points.append({
                    'position': point,
                    'normal': properties['normal'],
                    'curvature': properties['curvature'],
                    'surface_properties': properties
                })
        
        return contact_points
    
    def sample_surface_points(self, mesh, density):
        """Sample points on mesh surface"""
        # Use Poisson disk sampling or uniform sampling
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # For simplicity, return random vertex samples
        # In practice, use proper surface sampling
        n_samples = min(density, len(vertices))
        indices = np.random.choice(len(vertices), n_samples, replace=False)
        
        return vertices[indices]
    
    def analyze_surface_properties(self, mesh, point):
        """Analyze surface properties at given point"""
        # Calculate normal and curvature
        # This is a simplified implementation
        return {
            'normal': np.array([0.0, 0.0, 1.0]),  # Placeholder
            'curvature': 0.1,  # Placeholder
            'friction_coefficient': 0.8  # Typical for dry surfaces
        }
    
    def is_good_contact_point(self, properties):
        """Check if point is suitable for contact"""
        # Good contact points have moderate curvature and good friction
        return (properties['curvature'] < 10.0 and 
                properties['friction_coefficient'] > 0.3)

class StabilityEvaluator:
    """Evaluate grasp stability"""
    def check_force_closure(self, contact_points, object_properties):
        """Check if grasp achieves force closure"""
        # Force closure means the grasp can resist any external wrench
        # This is a simplified check
        if len(contact_points) < 2:
            return False
        
        # For two-point grasp, check antipodality
        if len(contact_points) == 2:
            return self.check_two_point_antipodality(contact_points)
        
        # For multi-point grasp, check force closure conditions
        return self.check_multi_point_force_closure(contact_points)
    
    def check_two_point_antipodality(self, contact_points):
        """Check if two-contact grasp is approximately antipodal"""
        p1, p2 = contact_points
        n1, n2 = p1['normal'], p2['normal']
        
        # Antipodal grasp: normals point roughly toward each other
        vector_between = p2['position'] - p1['position']
        normalized_vector = vector_between / np.linalg.norm(vector_between)
        
        # Check if normals align with the connection vector
        alignment1 = np.dot(n1, normalized_vector)
        alignment2 = np.dot(n2, -normalized_vector)
        
        # Both normals should point toward the other contact
        return alignment1 > 0.7 and alignment2 > 0.7

class GraspQualityMetrics:
    """Calculate various grasp quality metrics"""
    def evaluate_grasp(self, contact_pair, object_properties):
        """Evaluate quality of grasp"""
        contact1, contact2 = contact_pair
        
        # Calculate force closure quality
        force_closure_quality = self.calculate_force_closure_quality(contact_pair)
        
        # Calculate volume displacement (how well grasp centers the object)
        volume_disp = self.calculate_volume_displacement(contact_pair, object_properties)
        
        # Calculate contact surface alignment
        alignment = self.calculate_surface_alignment(contact_pair)
        
        # Check antipodality
        antipodal = self.check_antipodality(contact_pair)
        
        return {
            'force_closure': force_closure_quality > 0.5,
            'force_closure_quality': force_closure_quality,
            'volume_displacement': volume_disp,
            'contact_surface_alignment': alignment,
            'antipodal': antipodal,
            'overall_quality': (force_closure_quality + volume_disp + alignment) / 3.0
        }
    
    def calculate_force_closure_quality(self, contact_pair):
        """Calculate quantitative measure of force closure"""
        # This would involve calculating the grasp matrix and checking its properties
        # For now, return a simplified measure
        contact1, contact2 = contact_pair
        distance = np.linalg.norm(contact2['position'] - contact1['position'])
        
        # Optimal grasp width is related to object size
        optimal_width = 0.3  # Simplified
        width_factor = min(distance / optimal_width, optimal_width / distance)
        
        return width_factor
```

## Dual-Arm Coordination

### Bimanual Manipulation Strategies

Humanoid robots with two arms can perform complex bimanual tasks:

```python
# Dual-arm coordination system
class DualArmCoordinator:
    def __init__(self):
        self.left_arm = ArmController(side='left')
        self.right_arm = ArmController(side='right')
        self.coordination_planner = CoordinationPlanner()
        self.task_decomposer = TaskDecomposer()
    
    def execute_bimanual_task(self, task_description):
        """Execute bimanual task with coordinated arms"""
        # Decompose task into bimanual subtasks
        subtasks = self.task_decomposer.decompose(task_description)
        
        for subtask in subtasks:
            if subtask['type'] == 'bimanual':
                self.execute_coordinated_action(subtask)
            elif subtask['type'] == 'unimanual':
                arm = self.left_arm if subtask['arm'] == 'left' else self.right_arm
                arm.execute_action(subtask['action'])
    
    def execute_coordinated_action(self, subtask):
        """Execute action requiring both arms"""
        # Plan coordinated motion
        left_plan, right_plan = self.coordination_planner.plan_coordinated_motion(
            subtask['action'], subtask['constraints']
        )
        
        # Execute in coordination
        self.left_arm.execute_trajectory(left_plan)
        self.right_arm.execute_trajectory(right_plan)
        
        # Monitor coordination during execution
        self.monitor_coordination(left_plan, right_plan)
    
    def plan_bimanual_manipulation(self, object_info, task_type):
        """Plan bimanual manipulation strategy"""
        if task_type == 'lifting_large_object':
            return self.plan_lift_with_both_hands(object_info)
        elif task_type == 'opening_container':
            return self.plan_open_with_both_hands(object_info)
        elif task_type == 'assembly_task':
            return self.plan_assembly_with_both_hands(object_info)
        else:
            return self.plan_generic_bimanual_task(object_info)
    
    def plan_lift_with_both_hands(self, object_info):
        """Plan lifting with both hands"""
        # Calculate object center of mass
        com = object_info['center_of_mass']
        dimensions = object_info['dimensions']
        
        # Plan grasp points on opposite sides of COM
        left_grasp_pos = com + np.array([-dimensions[0]/2, 0, 0])
        right_grasp_pos = com + np.array([dimensions[0]/2, 0, 0])
        
        # Plan approach trajectories
        left_approach = self.left_arm.plan_approach(left_grasp_pos)
        right_approach = self.right_arm.plan_approach(right_grasp_pos)
        
        # Plan synchronized lift
        lift_trajectory = {
            'left': self.left_arm.plan_lift(left_grasp_pos),
            'right': self.right_arm.plan_lift(right_grasp_pos),
            'synchronization': 'tight'  # Both arms move together
        }
        
        return {
            'approach_trajectories': {
                'left': left_approach,
                'right': right_approach
            },
            'grasp_configuration': {
                'left': self.left_arm.plan_grasp(left_grasp_pos),
                'right': self.right_arm.plan_grasp(right_grasp_pos)
            },
            'lift_trajectory': lift_trajectory,
            'safety_constraints': self.calculate_safety_constraints()
        }

class CoordinationPlanner:
    def __init__(self):
        self.collision_checker = CollisionChecker()
        self.synchronizer = MotionSynchronizer()
    
    def plan_coordinated_motion(self, action, constraints):
        """Plan coordinated motion for both arms"""
        # Plan individual motions
        left_plan = self.plan_arm_motion('left', action, constraints)
        right_plan = self.plan_arm_motion('right', action, constraints)
        
        # Check for collisions between arms
        if self.collision_checker.check_interarm_collision(left_plan, right_plan):
            # Re-plan with collision avoidance
            left_plan, right_plan = self.resolve_collisions(left_plan, right_plan)
        
        # Synchronize motions if required
        if constraints.get('synchronized', False):
            left_plan, right_plan = self.synchronizer.synchronize(
                left_plan, right_plan
            )
        
        return left_plan, right_plan
    
    def plan_arm_motion(self, arm_side, action, constraints):
        """Plan motion for individual arm"""
        # This would call individual arm motion planner
        pass
    
    def resolve_collisions(self, left_plan, right_plan):
        """Resolve collisions between arms"""
        # Implement collision resolution strategies
        # Could involve retiming, replanning, or coordination constraints
        pass

class MotionSynchronizer:
    """Synchronize motion between arms"""
    def synchronize(self, left_plan, right_plan):
        """Synchronize two motion plans"""
        # Ensure both arms reach key waypoints at the same time
        max_duration = max(
            self.get_plan_duration(left_plan),
            self.get_plan_duration(right_plan)
        )
        
        # Adjust speeds to match durations
        adjusted_left = self.adjust_plan_timing(left_plan, max_duration)
        adjusted_right = self.adjust_plan_timing(right_plan, max_duration)
        
        return adjusted_left, adjusted_right
    
    def get_plan_duration(self, plan):
        """Get duration of motion plan"""
        # Calculate plan duration
        return 1.0  # Placeholder
    
    def adjust_plan_timing(self, plan, target_duration):
        """Adjust plan timing to match target duration"""
        # Scale plan timing
        return plan  # Placeholder
```

## Adaptive Grasping

### Learning-Based Grasp Adaptation

Adaptive grasping systems learn from experience to improve grasp success:

```python
# Adaptive grasping system
import torch
import torch.nn as nn
import numpy as np

class AdaptiveGraspLearner:
    def __init__(self):
        self.grasp_network = GraspQualityNetwork()
        self.experience_replay = ExperienceReplayBuffer()
        self.action_selector = GraspActionSelector()
        
    def learn_from_grasp_experience(self, state, grasp_action, outcome):
        """Learn from grasp attempt experience"""
        # Store experience
        experience = {
            'state': state,  # Object properties, environment state
            'action': grasp_action,  # Grasp configuration
            'outcome': outcome,  # Success/failure, grasp quality
            'features': self.extract_features(state, grasp_action)
        }
        
        self.experience_replay.add(experience)
        
        # Update network if enough experiences
        if self.experience_replay.can_train():
            batch = self.experience_replay.sample_batch()
            self.update_network(batch)
    
    def predict_grasp_quality(self, state, grasp_action):
        """Predict quality of proposed grasp"""
        features = self.extract_features(state, grasp_action)
        quality_prediction = self.grasp_network.predict(features)
        return quality_prediction
    
    def select_best_grasp(self, candidate_grasps, object_state):
        """Select best grasp from candidates using learned model"""
        best_grasp = None
        best_quality = -float('inf')
        
        for grasp in candidate_grasps:
            quality = self.predict_grasp_quality(object_state, grasp)
            if quality > best_quality:
                best_quality = quality
                best_grasp = grasp
        
        return best_grasp
    
    def extract_features(self, state, grasp_action):
        """Extract features for grasp quality prediction"""
        features = []
        
        # Object features
        features.extend([
            state['object_size'][0],  # Width
            state['object_size'][1],  # Height  
            state['object_size'][2],  # Depth
            state['object_weight'],
            state['object_com'][0],   # COM x
            state['object_com'][1],   # COM y
            state['object_com'][2],   # COM z
        ])
        
        # Grasp features
        features.extend([
            grasp_action['grasp_width'],
            grasp_action['grasp_angle'],
            grasp_action['contact_points'][0][0],  # Contact 1 x
            grasp_action['contact_points'][0][1],  # Contact 1 y
            grasp_action['contact_points'][1][0],  # Contact 2 x
            grasp_action['contact_points'][1][1],  # Contact 2 y
        ])
        
        # Surface properties
        features.extend([
            state['surface_friction'],
            state['surface_roughness'],
        ])
        
        return np.array(features)

class GraspQualityNetwork(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output: quality score
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, features):
        """Predict grasp quality for given features"""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            quality = self.forward(features_tensor)
            return quality.item()

class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, experience):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample_batch(self, batch_size=32):
        """Sample batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def can_train(self):
        """Check if buffer has enough experiences for training"""
        return len(self.buffer) > 1000

class GraspActionSelector:
    """Select grasp actions based on learned models"""
    def select_action(self, state, learned_model, exploration_rate=0.1):
        """Select grasp action with exploration"""
        if np.random.random() < exploration_rate:
            # Explore: random grasp
            return self.generate_random_grasp(state)
        else:
            # Exploit: best predicted grasp
            candidates = self.generate_grasp_candidates(state)
            return learned_model.select_best_grasp(candidates, state)
    
    def generate_grasp_candidates(self, state):
        """Generate candidate grasps for state"""
        # Generate multiple grasp configurations
        candidates = []
        
        # Vary grasp width
        for width in np.linspace(0.02, 0.08, 5):
            # Vary grasp angle
            for angle in np.linspace(-np.pi/4, np.pi/4, 5):
                candidates.append({
                    'grasp_width': width,
                    'grasp_angle': angle,
                    'contact_points': self.calculate_contact_points(state, width, angle)
                })
        
        return candidates
    
    def calculate_contact_points(self, state, width, angle):
        """Calculate contact points for grasp configuration"""
        # Simplified calculation
        center = state['object_com']
        direction = np.array([np.cos(angle), np.sin(angle), 0])
        
        point1 = center - direction * width/2
        point2 = center + direction * width/2
        
        return [point1, point2]

# Grasp adaptation during execution
class OnlineGraspAdapter:
    def __init__(self, grasp_controller):
        self.controller = grasp_controller
        self.sensors = self.initialize_sensors()
        self.adaptation_engine = AdaptationEngine()
    
    def execute_adaptive_grasp(self, object_info, initial_grasp):
        """Execute grasp with online adaptation"""
        # Execute initial grasp approach
        self.controller.move_to_pregrasp(initial_grasp['approach_pose'])
        
        # Monitor during approach
        feedback = self.monitor_approach(object_info)
        
        if feedback['needs_adjustment']:
            # Adapt grasp based on real-time feedback
            adapted_grasp = self.adaptation_engine.adapt_grasp(
                initial_grasp, feedback
            )
        else:
            adapted_grasp = initial_grasp
        
        # Execute adapted grasp
        self.controller.execute_grasp(adapted_grasp)
        
        # Monitor grasp quality during execution
        grasp_outcome = self.monitor_grasp_execution(adapted_grasp)
        
        return grasp_outcome
    
    def monitor_approach(self, object_info):
        """Monitor approach phase and detect issues"""
        # Use tactile, vision, and proprioceptive feedback
        tactile_data = self.sensors['tactile'].read()
        vision_data = self.sensors['camera'].capture()
        proprio_data = self.sensors['encoders'].read()
        
        feedback = {
            'object_detected': self.detect_object_proximity(vision_data),
            'slip_detected': self.detect_slip(tactile_data),
            'alignment_error': self.calculate_alignment_error(proprio_data, object_info),
            'needs_adjustment': False
        }
        
        # Determine if adjustment is needed
        if (feedback['alignment_error'] > 0.01 or  # 1cm error
            feedback['slip_detected']):
            feedback['needs_adjustment'] = True
        
        return feedback
    
    def monitor_grasp_execution(self, grasp_config):
        """Monitor grasp execution and assess quality"""
        # Monitor force, position, and tactile feedback
        force_data = self.sensors['force_torque'].read()
        position_data = self.sensors['encoders'].read()
        tactile_data = self.sensors['tactile'].read()
        
        # Assess grasp quality
        quality_metrics = {
            'grasp_force': np.mean(force_data['gripper']),
            'object_slip': self.detect_object_movement(),
            'contact_stability': self.assess_contact_stability(tactile_data),
            'success': self.assess_grasp_success(force_data, tactile_data)
        }
        
        return quality_metrics
```

## Whole-Body Manipulation

### Integrating Arms with Locomotion

Humanoid robots can enhance manipulation by coordinating arms with whole-body motion:

```python
# Whole-body manipulation controller
class WholeBodyManipulationController:
    def __init__(self):
        self.arm_controllers = {
            'left': ArmController('left'),
            'right': ArmController('right')
        }
        self.base_controller = BaseController()
        self.balance_controller = BalanceController()
        self.whole_body_planner = WholeBodyMotionPlanner()
    
    def execute_reaching_task(self, target_pose, constraints=None):
        """Execute reaching task using whole body"""
        # Plan whole-body motion to reach target
        motion_plan = self.whole_body_planner.plan_reaching_motion(
            target_pose, constraints
        )
        
        # Execute coordinated motion
        self.execute_whole_body_motion(motion_plan)
    
    def execute_manipulation_with_mobility(self, task_description):
        """Execute manipulation that may require mobility"""
        # Analyze task requirements
        task_analysis = self.analyze_manipulation_requirements(task_description)
        
        if task_analysis['reachable_standing']:
            # Execute with fixed base
            return self.execute_standing_manipulation(task_description)
        elif task_analysis['reachable_moving']:
            # Execute with base motion
            return self.execute_mobile_manipulation(task_description)
        else:
            # Need to reposition first
            return self.execute_repositioning_manipulation(task_description)
    
    def analyze_manipulation_requirements(self, task_description):
        """Analyze if task is reachable with current configuration"""
        # Determine workspace requirements
        required_workspace = self.calculate_required_workspace(task_description)
        
        # Check current reachability
        current_workspace = self.get_current_workspace()
        
        analysis = {
            'reachable_standing': self.is_within_workspace(
                required_workspace, current_workspace
            ),
            'reachable_moving': self.is_reachable_with_moving_base(
                required_workspace
            ),
            'workspace_overlap': self.calculate_workspace_overlap(
                required_workspace, current_workspace
            )
        }
        
        return analysis
    
    def execute_standing_manipulation(self, task_description):
        """Execute manipulation without base movement"""
        # Plan arm motion only
        arm_plan = self.whole_body_planner.plan_arm_only_motion(
            task_description
        )
        
        # Execute with balance maintenance
        self.balance_controller.activate_stance_mode()
        self.execute_arm_motion(arm_plan)
    
    def execute_mobile_manipulation(self, task_description):
        """Execute manipulation with coordinated base/arm motion"""
        # Plan coordinated base and arm motion
        coordinated_plan = self.whole_body_planner.plan_coordinated_motion(
            task_description
        )
        
        # Execute with dynamic balance
        self.balance_controller.activate_dynamic_mode()
        
        # Execute base and arm motions in coordination
        self.base_controller.execute_trajectory(
            coordinated_plan['base_trajectory']
        )
        self.execute_arm_motion(coordinated_plan['arm_trajectories'])
    
    def execute_repositioning_manipulation(self, task_description):
        """Execute manipulation requiring prior repositioning"""
        # First, plan navigation to better position
        navigation_goal = self.calculate_optimal_manipulation_position(
            task_description
        )
        
        # Navigate to position
        self.base_controller.navigate_to(navigation_goal)
        
        # Then execute manipulation from new position
        return self.execute_standing_manipulation(task_description)

class WholeBodyMotionPlanner:
    def __init__(self):
        self.arm_planner = ArmMotionPlanner()
        self.base_planner = BaseMotionPlanner()
        self.integration_planner = IntegrationPlanner()
    
    def plan_reaching_motion(self, target_pose, constraints):
        """Plan whole-body reaching motion"""
        # Try different strategies
        strategies = [
            self.plan_arm_only_reach,
            self.plan_arm_with_trunk,
            self.plan_full_body_reach
        ]
        
        for strategy in strategies:
            try:
                plan = strategy(target_pose, constraints)
                if self.validate_plan(plan):
                    return plan
            except PlanningException:
                continue
        
        # If no strategy works, raise exception
        raise PlanningException("No valid reaching plan found")
    
    def plan_arm_only_reach(self, target_pose, constraints):
        """Plan reaching with arms only"""
        left_plan = self.arm_planner.plan_motion(
            'left', target_pose, constraints
        )
        right_plan = self.arm_planner.plan_motion(
            'right', target_pose, constraints
        )
        
        return {
            'base_motion': [],  # No base motion
            'left_arm_motion': left_plan,
            'right_arm_motion': right_plan,
            'balance_constraints': self.calculate_balance_constraints()
        }
    
    def plan_arm_with_trunk(self, target_pose, constraints):
        """Plan reaching with arms and trunk"""
        # Plan coordinated arm and trunk motion
        # This would involve a more complex kinematic model
        pass
    
    def plan_full_body_reach(self, target_pose, constraints):
        """Plan reaching with full body coordination"""
        # Plan base, trunk, and arm motion together
        base_plan = self.base_planner.plan_approach(target_pose)
        arm_plans = self.plan_coordinated_arm_motion(target_pose, base_plan)
        
        return {
            'base_motion': base_plan,
            'arm_motion': arm_plans,
            'balance_trajectory': self.calculate_balance_trajectory(base_plan)
        }
    
    def validate_plan(self, plan):
        """Validate that plan satisfies all constraints"""
        # Check for collisions
        if self.has_collision(plan):
            return False
        
        # Check balance constraints
        if not self.is_balanced(plan):
            return False
        
        # Check joint limits
        if self.has_joint_limit_violation(plan):
            return False
        
        return True

class BalanceController:
    def __init__(self):
        self.balance_algorithm = self.initialize_balance_algorithm()
        self.sensors = self.initialize_balance_sensors()
    
    def activate_stance_mode(self):
        """Activate balance control for standing manipulation"""
        # Use simpler balance control
        self.balance_algorithm.set_mode('static')
    
    def activate_dynamic_mode(self):
        """Activate balance control for dynamic motion"""
        # Use more sophisticated balance control
        self.balance_algorithm.set_mode('dynamic')
    
    def calculate_balance_correction(self, current_state, desired_motion):
        """Calculate balance correction for desired motion"""
        # Use inverted pendulum model or similar
        com_position = self.sensors['imu'].get_com_position()
        com_velocity = self.sensors['imu'].get_com_velocity()
        
        # Calculate zero moment point (ZMP)
        zmp = self.calculate_zmp(com_position, com_velocity)
        
        # Determine balance correction
        correction = self.balance_algorithm.compute_correction(
            zmp, desired_motion
        )
        
        return correction
```

## Safety and Compliance

### Safe Grasping and Manipulation

Safety is paramount in humanoid manipulation systems:

```python
# Safety system for manipulation
class ManipulationSafetySystem:
    def __init__(self):
        self.force_limiter = ForceLimiter()
        self.collision_detector = CollisionDetector()
        self.emergency_handler = EmergencyHandler()
        self.compliance_controller = ComplianceController()
    
    def wrap_manipulation_action(self, action_function):
        """Wrap manipulation action with safety checks"""
        def safe_action(*args, **kwargs):
            # Pre-execution safety checks
            if not self.pre_execution_check(*args, **kwargs):
                raise SafetyException("Pre-execution safety check failed")
            
            try:
                # Execute with safety monitoring
                result = action_function(*args, **kwargs)
                
                # Post-execution validation
                if not self.post_execution_check(result):
                    raise SafetyException("Post-execution validation failed")
                
                return result
            except Exception as e:
                # Emergency handling
                self.emergency_handler.handle_exception(e)
                raise
        
        return safe_action
    
    def pre_execution_check(self, *args, **kwargs):
        """Perform safety checks before execution"""
        checks = [
            self.check_force_limits,
            self.check_collision_risk,
            self.check_workspace_boundaries,
            self.check_human_safety_zone
        ]
        
        return all(check() for check in checks)
    
    def post_execution_check(self, result):
        """Perform safety checks after execution"""
        # Verify successful completion
        # Check for unexpected forces
        # Verify object status
        return True  # Simplified
    
    def monitor_execution(self, action_generator):
        """Monitor action execution for safety violations"""
        for step in action_generator:
            # Check safety at each step
            if not self.step_safety_check(step):
                self.emergency_handler.trigger_safety_stop()
                return False
            yield step

class ForceLimiter:
    def __init__(self, max_gripper_force=50.0, max_wrist_force=100.0):
        self.max_gripper_force = max_gripper_force
        self.max_wrist_force = max_wrist_force
    
    def limit_force(self, desired_force, sensor_feedback):
        """Limit force based on safety thresholds"""
        current_force = sensor_feedback.get('gripper_force', 0)
        
        if current_force > self.max_gripper_force:
            # Reduce commanded force
            reduction_factor = self.max_gripper_force / current_force
            limited_force = desired_force * reduction_factor
            return limited_force
        
        return desired_force
    
    def check_force_safety(self, force_readings):
        """Check if force readings are within safe limits"""
        gripper_force = force_readings.get('gripper', 0)
        wrist_force = force_readings.get('wrist', np.zeros(6))
        
        return (gripper_force <= self.max_gripper_force and
                np.linalg.norm(wrist_force) <= self.max_wrist_force)

class ComplianceController:
    def __init__(self):
        self.stiffness_scheduler = StiffnessScheduler()
        self.impedance_controller = ImpedanceController()
    
    def execute_compliant_manipulation(self, trajectory, compliance_requirements):
        """Execute manipulation with variable compliance"""
        compliant_trajectory = []
        
        for waypoint in trajectory:
            # Adjust impedance based on task requirements
            impedance_params = self.stiffness_scheduler.schedule(
                waypoint, compliance_requirements
            )
            
            # Execute with impedance control
            compliant_waypoint = self.impedance_controller.track_waypoint(
                waypoint, impedance_params
            )
            
            compliant_trajectory.append(compliant_waypoint)
        
        return compliant_trajectory

class StiffnessScheduler:
    """Schedule variable stiffness based on task requirements"""
    def schedule(self, waypoint, requirements):
        """Schedule stiffness for given waypoint"""
        if requirements.get('delicate', False):
            return {'stiffness': 100, 'damping': 10}  # Low stiffness
        elif requirements.get('firm', False):
            return {'stiffness': 1000, 'damping': 100}  # High stiffness
        else:
            return {'stiffness': 500, 'damping': 50}  # Medium stiffness

class ImpedanceController:
    """Impedance control for compliant manipulation"""
    def track_waypoint(self, desired_pose, impedance_params):
        """Track waypoint with impedance control"""
        # Implement impedance control law
        # F = k(x_d - x) + b(v_d - v)
        pass
```

## Best Practices

### Manipulation Design Principles
1. **Redundancy**: Use redundant degrees of freedom for robust manipulation
2. **Adaptability**: Design systems that adapt to object variations
3. **Safety**: Prioritize safety in all manipulation actions
4. **Efficiency**: Optimize for energy and time efficiency
5. **Learning**: Incorporate learning from experience

### Implementation Guidelines
1. **Modular Architecture**: Separate grasp planning, motion planning, and control
2. **Real-Time Capability**: Ensure real-time response for safety
3. **Sensor Integration**: Combine multiple sensory modalities
4. **Uncertainty Handling**: Account for uncertainty in perception and control
5. **Human-Robot Safety**: Design for safe human-robot interaction

## Troubleshooting Common Issues

### Grasp Failures
- **Slippage**: Increase grasp force or improve contact geometry
- **Misalignment**: Improve visual servoing and approach planning
- **Inadequate Force**: Adjust force control parameters
- **Object Damage**: Implement compliance and force limiting

### Coordination Problems
- **Timing Issues**: Improve synchronization between arms
- **Collision Conflicts**: Enhance collision detection and avoidance
- **Balance Problems**: Strengthen whole-body control integration
- **Workspace Limitations**: Plan coordinated base/arm motions

## Summary

In this chapter, we've explored advanced manipulation and grasping strategies for humanoid robots. We've covered anthropomorphic hand design, geometric grasp planning, dual-arm coordination, adaptive grasping, and whole-body manipulation. Effective humanoid manipulation requires integrating multiple systems and considering the unique challenges of human-like dexterity in real-world environments.