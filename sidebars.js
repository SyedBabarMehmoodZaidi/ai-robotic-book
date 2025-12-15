// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/index',
        'module-1-ros2/architecture',
        'module-1-ros2/python-integration',
        'module-1-ros2/urdf-robots',
        'module-1-ros2/packages-management',
        'module-1-ros2/exercises'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/index',
        'module-2-digital-twin/gazebo-setup',
        'module-2-digital-twin/unity-simulation',
        'module-2-digital-twin/urdf-sdf-formats',
        'module-2-digital-twin/sensor-simulation',
        'module-2-digital-twin/exercises'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-ai-robot-brain/index',
        'module-3-ai-robot-brain/isaac-sim-setup',
        'module-3-ai-robot-brain/isaac-ros-vslam',
        'module-3-ai-robot-brain/nav2-path-planning',
        'module-3-ai-robot-brain/reinforcement-learning-locomotion',
        'module-3-ai-robot-brain/exercises'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/index',
        'module-4-vla/vla-architecture',
        'module-4-vla/multimodal-perception',
        'module-4-vla/language-guided-manipulation',
        'module-4-vla/vla-system-integration',
        'module-4-vla/exercises'
      ],
    },
    'capstone-project',
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/hardware-requirements',
        'appendices/installation-guide',
        'appendices/troubleshooting'
      ],
    },
    {
      type: 'category',
      label: 'RAG System Contracts',
      items: [
        'contracts/rag-api-contracts',
        'contracts/rag-data-models',
        'contracts/rag-error-handling',
        'contracts/rag-performance-optimization',
        'contracts/rag-security-implementation',
        'contracts/rag-configuration-management'
      ],
    },
    {
      type: 'category',
      label: 'User Guides',
      items: [
        'user-guide/rag-features-user-guide'
      ],
    },
    {
      type: 'category',
      label: 'Developer Guides',
      items: [
        'developer-guide/rag-developer-documentation'
      ],
    },
    'references',
    'glossary',
    'conclusion'
  ],
};

module.exports = sidebars;