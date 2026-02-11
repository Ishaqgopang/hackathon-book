# Physical AI & Humanoid Robotics Course - Frontend

This directory contains the Docusaurus-based frontend for the Physical AI & Humanoid Robotics course book.

## Installation

```bash
npm install
```

## Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:
```bash
USE_SSH=true npm run deploy
```

Not using SSH:
```bash
GH_TOKEN=<GITHUB_TOKEN> npm run deploy
```

For more details on deployment, see the [Docusaurus deployment guide](https://docusaurus.io/docs/deployment).

## Course Structure

The course is divided into four main modules:

- **Module 1**: The Robotic Nervous System (ROS 2)
- **Module 2**: The Digital Twin (Gazebo & Unity)
- **Module 3**: The AI-Robot Brain (NVIDIA Isaac™)
- **Module 4**: Vision-Language-Action & Autonomous Humanoid Capstone