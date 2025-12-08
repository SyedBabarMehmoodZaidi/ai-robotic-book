import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/ai-robotic-book/__docusaurus/debug',
    component: ComponentCreator('/ai-robotic-book/__docusaurus/debug', '7cb'),
    exact: true
  },
  {
    path: '/ai-robotic-book/__docusaurus/debug/config',
    component: ComponentCreator('/ai-robotic-book/__docusaurus/debug/config', 'ef1'),
    exact: true
  },
  {
    path: '/ai-robotic-book/__docusaurus/debug/content',
    component: ComponentCreator('/ai-robotic-book/__docusaurus/debug/content', 'aae'),
    exact: true
  },
  {
    path: '/ai-robotic-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/ai-robotic-book/__docusaurus/debug/globalData', 'd19'),
    exact: true
  },
  {
    path: '/ai-robotic-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/ai-robotic-book/__docusaurus/debug/metadata', '359'),
    exact: true
  },
  {
    path: '/ai-robotic-book/__docusaurus/debug/registry',
    component: ComponentCreator('/ai-robotic-book/__docusaurus/debug/registry', '1bc'),
    exact: true
  },
  {
    path: '/ai-robotic-book/__docusaurus/debug/routes',
    component: ComponentCreator('/ai-robotic-book/__docusaurus/debug/routes', '0c8'),
    exact: true
  },
  {
    path: '/ai-robotic-book/markdown-page',
    component: ComponentCreator('/ai-robotic-book/markdown-page', '6d4'),
    exact: true
  },
  {
    path: '/ai-robotic-book/docs',
    component: ComponentCreator('/ai-robotic-book/docs', 'c6b'),
    routes: [
      {
        path: '/ai-robotic-book/docs',
        component: ComponentCreator('/ai-robotic-book/docs', '493'),
        routes: [
          {
            path: '/ai-robotic-book/docs',
            component: ComponentCreator('/ai-robotic-book/docs', '2fb'),
            routes: [
              {
                path: '/ai-robotic-book/docs/appendices/hardware-requirements',
                component: ComponentCreator('/ai-robotic-book/docs/appendices/hardware-requirements', '724'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/appendices/installation-guide',
                component: ComponentCreator('/ai-robotic-book/docs/appendices/installation-guide', 'a2f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/appendices/troubleshooting',
                component: ComponentCreator('/ai-robotic-book/docs/appendices/troubleshooting', '363'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/capstone-project',
                component: ComponentCreator('/ai-robotic-book/docs/capstone-project', 'b0c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/conclusion',
                component: ComponentCreator('/ai-robotic-book/docs/conclusion', 'b0b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/glossary',
                component: ComponentCreator('/ai-robotic-book/docs/glossary', '974'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/intro',
                component: ComponentCreator('/ai-robotic-book/docs/intro', '427'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-1-ros2/',
                component: ComponentCreator('/ai-robotic-book/docs/module-1-ros2/', '3b3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-1-ros2/architecture',
                component: ComponentCreator('/ai-robotic-book/docs/module-1-ros2/architecture', '8f6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-1-ros2/exercises',
                component: ComponentCreator('/ai-robotic-book/docs/module-1-ros2/exercises', '02c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-1-ros2/packages-management',
                component: ComponentCreator('/ai-robotic-book/docs/module-1-ros2/packages-management', '908'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-1-ros2/python-integration',
                component: ComponentCreator('/ai-robotic-book/docs/module-1-ros2/python-integration', 'b6c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-1-ros2/urdf-robots',
                component: ComponentCreator('/ai-robotic-book/docs/module-1-ros2/urdf-robots', '8f4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-2-digital-twin/',
                component: ComponentCreator('/ai-robotic-book/docs/module-2-digital-twin/', 'c9c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-2-digital-twin/exercises',
                component: ComponentCreator('/ai-robotic-book/docs/module-2-digital-twin/exercises', '63c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-2-digital-twin/gazebo-setup',
                component: ComponentCreator('/ai-robotic-book/docs/module-2-digital-twin/gazebo-setup', 'dfd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-2-digital-twin/sensor-simulation',
                component: ComponentCreator('/ai-robotic-book/docs/module-2-digital-twin/sensor-simulation', '7b2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-2-digital-twin/unity-simulation',
                component: ComponentCreator('/ai-robotic-book/docs/module-2-digital-twin/unity-simulation', 'd40'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-2-digital-twin/urdf-sdf-formats',
                component: ComponentCreator('/ai-robotic-book/docs/module-2-digital-twin/urdf-sdf-formats', 'f47'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-3-ai-robot-brain/',
                component: ComponentCreator('/ai-robotic-book/docs/module-3-ai-robot-brain/', '5d1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-3-ai-robot-brain/exercises',
                component: ComponentCreator('/ai-robotic-book/docs/module-3-ai-robot-brain/exercises', '7ce'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-3-ai-robot-brain/isaac-ros-vslam',
                component: ComponentCreator('/ai-robotic-book/docs/module-3-ai-robot-brain/isaac-ros-vslam', '5eb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-3-ai-robot-brain/isaac-sim-setup',
                component: ComponentCreator('/ai-robotic-book/docs/module-3-ai-robot-brain/isaac-sim-setup', '498'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-3-ai-robot-brain/nav2-path-planning',
                component: ComponentCreator('/ai-robotic-book/docs/module-3-ai-robot-brain/nav2-path-planning', 'ade'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-3-ai-robot-brain/reinforcement-learning-locomotion',
                component: ComponentCreator('/ai-robotic-book/docs/module-3-ai-robot-brain/reinforcement-learning-locomotion', '79a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-4-vla/',
                component: ComponentCreator('/ai-robotic-book/docs/module-4-vla/', '6c1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-4-vla/exercises',
                component: ComponentCreator('/ai-robotic-book/docs/module-4-vla/exercises', '2c9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-4-vla/language-guided-manipulation',
                component: ComponentCreator('/ai-robotic-book/docs/module-4-vla/language-guided-manipulation', '26f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-4-vla/multimodal-perception',
                component: ComponentCreator('/ai-robotic-book/docs/module-4-vla/multimodal-perception', '03d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-4-vla/vla-architecture',
                component: ComponentCreator('/ai-robotic-book/docs/module-4-vla/vla-architecture', 'ac8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/module-4-vla/vla-system-integration',
                component: ComponentCreator('/ai-robotic-book/docs/module-4-vla/vla-system-integration', 'b19'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-robotic-book/docs/references',
                component: ComponentCreator('/ai-robotic-book/docs/references', '515'),
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
    path: '/ai-robotic-book/',
    component: ComponentCreator('/ai-robotic-book/', '729'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
