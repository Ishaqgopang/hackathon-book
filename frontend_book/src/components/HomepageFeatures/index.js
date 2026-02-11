import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '🤖 Physical AI & Humanoid Robotics',
    icon: '🤖',
    description: (
      <>
        Master the art of building intelligent humanoid robots using cutting-edge technologies
        including ROS 2, Digital Twins, and NVIDIA Isaac platforms. Learn to create robots
        that think, perceive, and act in the physical world.
      </>
    ),
  },
  {
    title: '🧩 Modular Learning Approach',
    icon: '🧩',
    description: (
      <>
        Comprehensive modules covering ROS 2 foundations, Digital Twin simulation,
        AI perception with NVIDIA Isaac, and Vision-Language-Action systems.
        Each module builds upon the previous, creating a complete learning pathway.
      </>
    ),
  },
  {
    title: '🏭 Industry-Ready Curriculum',
    icon: '🏭',
    description: (
      <>
        Designed with industry standards and best practices to prepare students
        for careers in robotics and AI development. Real-world projects and
        hands-on experience with professional tools.
      </>
    ),
  },
  {
    title: '🚀 Advanced Technologies',
    icon: '🚀',
    description: (
      <>
        Dive deep into NVIDIA Isaac, Gazebo simulation, Unity integration,
        and state-of-the-art computer vision. Learn the tools that power
        the next generation of robotics applications.
      </>
    ),
  },
  {
    title: '🧠 AI-Powered Systems',
    icon: '🧠',
    description: (
      <>
        Explore how artificial intelligence enables robots to perceive, reason,
        and interact with complex environments. From computer vision to
        natural language processing and decision making.
      </>
    ),
  },
  {
    title: '🌐 Real-World Applications',
    icon: '🌐',
        description: (
      <>
        Apply your knowledge to real-world scenarios including autonomous
        navigation, human-robot interaction, manipulation tasks, and
        collaborative robotics in industrial and service environments.
      </>
    ),
  },
];

function Feature({icon, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.iconContainer}>
          <span className={styles.featureIcon}>{icon}</span>
        </div>
        <div className={styles.textContent}>
          <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
          <p className={styles.featureDescription}>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <div className="text--center padding-horiz--md">
          <h2 className={styles.sectionTitle}>What You'll Learn</h2>
          <p className={styles.sectionSubtitle}>Comprehensive curriculum designed for the future of robotics</p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}