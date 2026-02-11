/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 Foundations',
      items: [
        'module-1/intro',
        'module-1/chapter-1-ros2-fundamentals',
        'module-1/chapter-2-nodes-topics-services',
        'module-1/chapter-3-humanoid-modeling'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin',
      items: [
        'module-2/intro',
        'module-2/chapter-1-digital-twins-overview',
        'module-2/chapter-2-gazebo-physics',
        'module-2/chapter-3-environment-design',
        'module-2/chapter-4-unity-visualization',
        'module-2/chapter-5-sensor-simulation'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI Perception with NVIDIA Isaac',
      items: [
        'module-3/intro',
        'module-3/chapter-1-isaac-overview',
        'module-3/chapter-2-isaac-sim',
        'module-3/chapter-3-isaac-ros',
        'module-3/chapter-4-nav2-navigation',
        'module-3/chapter-5-sim-to-real'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      items: [
        'module-4/intro',
        'module-4/chapter-1-vision-language-action',
        'module-4/chapter-2',
        'module-4/chapter-3',
        'module-4/chapter-4',
        'module-4/chapter-5'
      ],
    }
  ],
};

export default sidebars;