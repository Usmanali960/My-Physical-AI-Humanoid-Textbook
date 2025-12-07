import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      collapsible: true,
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Module 01 - ROS 2: The Robotic Nervous System',
          collapsible: true,
          collapsed: false,
          items: [
            'physical-ai/module-01-chapter-01',
            'physical-ai/module-01-chapter-02',
            'physical-ai/module-01-chapter-03',
            'physical-ai/module-01-chapter-04'
          ],
        },
        {
          type: 'category',
          label: 'Module 02 - Digital Twin: Gazebo & Unity',
          collapsible: true,
          collapsed: true,
          items: [
            'physical-ai/module-02-chapter-01',
            'physical-ai/module-02-chapter-02',
            'physical-ai/module-02-chapter-03',
            'physical-ai/module-02-chapter-04'
          ],
        },
        {
          type: 'category',
          label: 'Module 03 - AI and Machine Learning Integration',
          collapsible: true,
          collapsed: true,
          items: [
            'physical-ai/module-03-chapter-01',
            'physical-ai/module-03-chapter-02',
            'physical-ai/module-03-chapter-03',
            'physical-ai/module-03-chapter-04'
          ],
        },
        {
          type: 'category',
          label: 'Module 04 - Sensor Systems and Perception',
          collapsible: true,
          collapsed: true,
          items: [
            'physical-ai/module-04-chapter-01',
            'physical-ai/module-04-chapter-02',
            'physical-ai/module-04-chapter-03',
            'physical-ai/module-04-chapter-04'
          ],
        },
        {
          type: 'category',
          label: 'Module 05 - Humanoid Robotics',
          collapsible: true,
          collapsed: true,
          items: [
            'physical-ai/module-05-chapter-01',
            'physical-ai/module-05-chapter-02',
            'physical-ai/module-05-chapter-03',
            'physical-ai/module-05-chapter-04'
          ],
        },
        {
          type: 'category',
          label: 'Module 06 - Capstone Project: Autonomous Humanoid',
          collapsible: true,
          collapsed: true,
          items: [
            'physical-ai/module-06-chapter-01',
            'physical-ai/module-06-chapter-02',
            'physical-ai/module-06-chapter-03',
            'physical-ai/module-06-chapter-04'
          ],
        },
      ],
    },
  ],
};

export default sidebars;
