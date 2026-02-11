import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './RoboticsConcept.module.css';

const RoboticsConcept = ({ title, description, icon, children }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Animation when component mounts
    const timer = setTimeout(() => setIsVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={clsx(styles.roboticsConcept, isVisible && styles.visible)}>
      <div className={styles.conceptHeader}>
        <div className={styles.conceptIcon}>
          {icon}
        </div>
        <h3 className={styles.conceptTitle}>{title}</h3>
      </div>
      <div className={styles.conceptContent}>
        <p className={styles.conceptDescription}>{description}</p>
        {children && <div className={styles.conceptDetails}>{children}</div>}
      </div>
    </div>
  );
};

export default RoboticsConcept;