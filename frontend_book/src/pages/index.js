import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroText}>
            <div className={clsx(styles.badge, styles.roboticsBadge)}>🤖 AI & Robotics</div>
            <Heading as="h1" className={clsx("hero__title", styles.heroTitle)}>
              {siteConfig.title}
            </Heading>
            <p className={clsx("hero__subtitle", styles.heroSubtitle)}>
              {siteConfig.tagline}
            </p>
            <div className={styles.heroButtons}>
              <Link
                className="button button--primary button--lg"
                to="/docs/intro">
                🚀 Start Learning
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs/module-1/intro">
                📚 Explore Modules
              </Link>
            </div>
          </div>
          <div className={styles.heroVisual}>
            <div className={styles.robotIcon}>
              <svg viewBox="0 0 100 100" className={styles.robotSvg}>
                <circle cx="50" cy="30" r="15" fill="var(--ifm-color-primary)" opacity="0.8"/>
                <rect x="35" y="45" width="30" height="30" rx="5" fill="var(--ifm-color-primary)" opacity="0.9"/>
                <rect x="25" y="50" width="10" height="20" fill="var(--ifm-color-secondary)" opacity="0.7"/>
                <rect x="65" y="50" width="10" height="20" fill="var(--ifm-color-secondary)" opacity="0.7"/>
                <rect x="40" y="75" width="8" height="15" fill="var(--ifm-color-accent)" opacity="0.6"/>
                <rect x="52" y="75" width="8" height="15" fill="var(--ifm-color-accent)" opacity="0.6"/>
                <circle cx="43" cy="28" r="2" fill="white"/>
                <circle cx="57" cy="28" r="2" fill="white"/>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function StatsSection() {
  return (
    <div className={styles.statsSection}>
      <div className="container">
        <div className={styles.statsGrid}>
          <div className={styles.statItem}>
            <div className={styles.statNumber}>4</div>
            <div className={styles.statLabel}>Learning Modules</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statNumber}>15+</div>
            <div className={styles.statLabel}>Detailed Chapters</div>
          </div>
          <div className={styles.statItem}>
            <div className={styles.statNumber}>∞</div>
            <div className={styles.statLabel}>Possibilities</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Building Intelligent Humanoid Robots with ROS 2, Digital Twins, and NVIDIA Isaac">
      <HomepageHeader />
      <StatsSection />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}