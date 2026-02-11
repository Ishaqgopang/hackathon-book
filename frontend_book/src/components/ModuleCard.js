import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './ModuleCard.module.css';

const ModuleCard = ({ module }) => {
  const {
    number,
    title,
    description,
    chapters,
    progress = 0,
    colorClass = '',
    link = '#'
  } = module;

  return (
    <Link to={link} className={clsx(styles.moduleCardLink)}>
      <div className={clsx(styles.moduleCard, colorClass)}>
        <div className={styles.moduleHeader}>
          <div className={styles.moduleBadge}>
            <span className={styles.moduleNumber}>Module {number}</span>
          </div>
          <h3 className={styles.moduleTitle}>{title}</h3>
        </div>
        
        <p className={styles.moduleDescription}>{description}</p>
        
        <div className={styles.moduleStats}>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>{chapters.length}</span>
            <span className={styles.statLabel}>Chapters</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>{progress}%</span>
            <span className={styles.statLabel}>Complete</span>
          </div>
        </div>
        
        <div className={styles.progressBar}>
          <div 
            className={styles.progressFill} 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        
        <div className={styles.chaptersPreview}>
          <h4 className={styles.chaptersTitle}>Chapters:</h4>
          <ul className={styles.chaptersList}>
            {chapters.slice(0, 3).map((chapter, index) => (
              <li key={index} className={styles.chapterItem}>
                {chapter}
              </li>
            ))}
            {chapters.length > 3 && (
              <li className={styles.chapterItem}>
                ... and {chapters.length - 3} more
              </li>
            )}
          </ul>
        </div>
      </div>
    </Link>
  );
};

export default ModuleCard;