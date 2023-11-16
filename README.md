# WaymaxRL

## Project Overview

This GitHub project presents a comprehensive reinforcement learning environment and training pipeline, meticulously designed to integrate with the [Waymax](https://github.com/waymo-research/waymax) simulator, which is based on the esteemed Waymo dataset. Our approach is heavily influenced by the principles and methodologies found in [Brax](https://github.com/google/brax), a renowned differentiable physics engine. This project focuses on implementing a versatile array of learning algorithms.

## Key Features

- Integration with Waymax Simulator: Utilizes the lightweight, multi-agent Waymax simulator, tailored for autonomous driving research.
- Support for Various RL Algorithms: Offers a rich set of learning algorithms, ensuring a comprehensive approach to reinforcement learning.
- Brax-Inspired Design: Leverages the efficiency and scalability of Brax for high-performance simulation and research applicability.

## Waymax Simulator

Waymax is a state-of-the-art, JAX-based simulator for autonomous driving research, drawing on the rich data of the Waymo Open Motion Dataset. It provides a simplified yet effective environment for behavior research in autonomous driving, encompassing aspects from closed-loop simulation for planning to open-loop behavior prediction. Key aspects include:

- Multi-Agent Capability: Supports research involving multiple agents, crucial for realistic autonomous driving scenarios.
- Bounding Box Representation: Simplifies complex environments by representing objects like vehicles and pedestrians as bounding boxes.
- JAX-Based Design: Ensures compatibility with modern, acceleration hardware-focused research workflows.

## Brax: A Source of Inspiration

Brax serves as a core inspiration for our project's architecture. It is a fast, fully differentiable physics engine widely used in various domains like robotics and reinforcement learning. Its features include:

- Written in JAX: Compatible with modern machine learning and simulation frameworks.
- Efficient and Scalable: Optimized for both single-device and massively parallel simulations.
- Versatile Application: Suitable for a wide range of simulation-heavy applications beyond reinforcement learning.

## Getting Started

Clone the Repository: 
```
git clone https://github.com/charraut/waymax-rl
```
Install Dependencies: Follow the installation guide provided in the documentation.
```
pip install -e .
```



