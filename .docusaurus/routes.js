import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '35b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '176'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'b49'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', '746'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '6ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '669'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '5ce'),
    exact: true
  },
  {
    path: '/markdown-page',
    component: ComponentCreator('/markdown-page', 'e16'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '066'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '947'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'c91'),
            routes: [
              {
                path: '/docs/appendices/hardware-requirements',
                component: ComponentCreator('/docs/appendices/hardware-requirements', '7d2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendices/installation-guide',
                component: ComponentCreator('/docs/appendices/installation-guide', '052'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendices/troubleshooting',
                component: ComponentCreator('/docs/appendices/troubleshooting', '11d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/capstone-project',
                component: ComponentCreator('/docs/capstone-project', '44f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/conclusion',
                component: ComponentCreator('/docs/conclusion', '187'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/glossary',
                component: ComponentCreator('/docs/glossary', 'be2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', 'aed'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/',
                component: ComponentCreator('/docs/module-1-ros2/', 'fb1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/architecture',
                component: ComponentCreator('/docs/module-1-ros2/architecture', 'e2c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/exercises',
                component: ComponentCreator('/docs/module-1-ros2/exercises', 'bf4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/packages-management',
                component: ComponentCreator('/docs/module-1-ros2/packages-management', '7b8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/python-integration',
                component: ComponentCreator('/docs/module-1-ros2/python-integration', '7c6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/urdf-robots',
                component: ComponentCreator('/docs/module-1-ros2/urdf-robots', '00a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/',
                component: ComponentCreator('/docs/module-2-digital-twin/', '1dd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/exercises',
                component: ComponentCreator('/docs/module-2-digital-twin/exercises', 'ffc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/gazebo-setup',
                component: ComponentCreator('/docs/module-2-digital-twin/gazebo-setup', '095'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/sensor-simulation',
                component: ComponentCreator('/docs/module-2-digital-twin/sensor-simulation', 'cbc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/unity-simulation',
                component: ComponentCreator('/docs/module-2-digital-twin/unity-simulation', '5fd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/urdf-sdf-formats',
                component: ComponentCreator('/docs/module-2-digital-twin/urdf-sdf-formats', 'dee'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-robot-brain/',
                component: ComponentCreator('/docs/module-3-ai-robot-brain/', 'ee8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-robot-brain/exercises',
                component: ComponentCreator('/docs/module-3-ai-robot-brain/exercises', 'e22'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-robot-brain/isaac-ros-vslam',
                component: ComponentCreator('/docs/module-3-ai-robot-brain/isaac-ros-vslam', 'c8b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-robot-brain/isaac-sim-setup',
                component: ComponentCreator('/docs/module-3-ai-robot-brain/isaac-sim-setup', '134'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-robot-brain/nav2-path-planning',
                component: ComponentCreator('/docs/module-3-ai-robot-brain/nav2-path-planning', '262'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-ai-robot-brain/reinforcement-learning-locomotion',
                component: ComponentCreator('/docs/module-3-ai-robot-brain/reinforcement-learning-locomotion', '4e7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/',
                component: ComponentCreator('/docs/module-4-vla/', '375'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/exercises',
                component: ComponentCreator('/docs/module-4-vla/exercises', 'cf9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/language-guided-manipulation',
                component: ComponentCreator('/docs/module-4-vla/language-guided-manipulation', 'bd5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/multimodal-perception',
                component: ComponentCreator('/docs/module-4-vla/multimodal-perception', 'c6d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/vla-architecture',
                component: ComponentCreator('/docs/module-4-vla/vla-architecture', '214'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/vla-system-integration',
                component: ComponentCreator('/docs/module-4-vla/vla-system-integration', '478'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/references',
                component: ComponentCreator('/docs/references', 'd1a'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '896'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
