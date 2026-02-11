import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import ModuleCard from '@site/src/components/ModuleCard';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Heading from '@theme/Heading';
import styles from './modules.module.css';

function ModulesOverview() {
  const {siteConfig} = useDocusaurusContext();

  const modules = [
    {
      number: 1,
      title: 'ROS 2 Foundations',
      description: 'Learn the fundamentals of ROS 2, the communication backbone of modern robotics. Understand nodes, topics, services, and how to model and control humanoid robots.',
      chapters: [
        'Introduction to ROS 2 and the Robotic Nervous System',
        'ROS 2 Communication: Nodes, Topics, and Services',
        'Humanoid Modeling with URDF and Python Control'
      ],
      progress: 100,
      colorClass: styles.module1,
      link: '/docs/module-1/intro'
    },
    {
      number: 2,
      title: 'Digital Twin',
      description: 'Explore physics-based simulation and digital twin concepts. Learn how to create realistic environments for testing and training humanoid robots using Gazebo and Unity.',
      chapters: [
        'Digital Twins and Simulation in Physical AI',
        'Physics Simulation with Gazebo (Gravity, Collisions, Dynamics)',
        'Environment Design and World Building in Gazebo',
        'High-Fidelity Interaction and Visualization with Unity',
        'Sensor Simulation: LiDAR, Depth Cameras, and IMUs'
      ],
      progress: 100,
      colorClass: styles.module2,
      link: '/docs/module-2/intro'
    },
    {
      number: 3,
      title: 'AI Perception with NVIDIA Isaac',
      description: 'Discover how to build intelligent perception and navigation systems using NVIDIA Isaac tools. Learn about photorealistic simulation, synthetic data generation, and accelerated perception pipelines.',
      chapters: [
        'NVIDIA Isaac Overview - AI-Powered Robotics Platform',
        'Isaac Sim - Advanced Simulation for Humanoid Robots',
        'Isaac ROS - GPU-Accelerated Perception Pipelines',
        'Navigation and Planning with Isaac and Nav2',
        'Sim-to-Real Transfer - Bridging Simulation and Reality'
      ],
      progress: 100,
      colorClass: styles.module3,
      link: '/docs/module-3/intro'
    },
    {
      number: 4,
      title: 'Vision-Language-Action',
      description: 'Combine everything learned to create an autonomous humanoid system capable of understanding and interacting with the world through vision, language, and action.',
      chapters: [
        'Vision-Language-Action Integration for Humanoid Robots',
        'Embodied AI: Reasoning and Physical Interaction',
        'Humanoid Manipulation and Grasping Strategies',
        'Social Interaction and Human-Robot Communication',
        'Autonomous Humanoid System Integration and Validation'
      ],
      progress: 100,
      colorClass: styles.module4,
      link: '/docs/module-4/intro'
    }
  ];

  return (
    <Layout
      title={`Learning Modules | ${siteConfig.title}`}
      description="Explore the comprehensive modules of the Physical AI & Humanoid Robotics course">
      <div className={styles.modulesHero}>
        <div className="container">
          <div className={styles.heroContent}>
            <div className={styles.heroText}>
              <div className={clsx(styles.badge, styles.courseBadge)}>📚 Complete Curriculum</div>
              <Heading as="h1" className={styles.heroTitle}>
                Learning Modules
              </Heading>
              <p className={styles.heroSubtitle}>
                A comprehensive journey through the essential topics of humanoid robotics, 
                from foundational concepts to advanced AI integration.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.modulesGrid}>
        <div className="container">
          <div className={styles.modulesContainer}>
            {modules.map((module, index) => (
              <ModuleCard key={index} module={module} />
            ))}
          </div>
        </div>
      </div>

      <div className={styles.ctaSection}>
        <div className="container">
          <div className={styles.ctaContent}>
            <Heading as="h2" className={styles.ctaTitle}>
              Ready to Start Your Journey?
            </Heading>
            <p className={styles.ctaSubtitle}>
              Begin with Module 1 to establish the foundational concepts that will be built upon throughout the course.
            </p>
            <Link
              className="button button--primary button--lg"
              to="/docs/intro">
              Start Learning
            </Link>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default ModulesOverview;