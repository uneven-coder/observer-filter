# Persistence Evolution Simulation

## Overview

This project explores the idea of **persistence**: the concept that existence itself acts as a filter. Consciousness is only experienced from surviving perspectives because non surviving states are unable to observe themselves.

![image of emergent agents](https://github.com/uneven-coder/black-wall-ai/blob/cd14769227987188e69925277d807546718a3ce9/assets/Screenshot%202026-05-14%20223839.png)

To experiment with this idea, I built a minimal evolutionary AI simulation where agents are not directly rewarded or punished. Instead, survival itself becomes the selection mechanism. Behaviours that allow agents to persist naturally propagate through mutation and inheritance over generations.

The goal of the project is not to create an optimal reinforcement learning system, but to observe how simple environmental pressure and persistence can produce increasingly stable behaviour over time.

![Emergent Agents](https://github.com/uneven-coder/black-wall-ai/blob/9b2efcbf75761803a803a14916586a5bb49a0ba4/assets/Recording%202026-05-15%20000504.gif)

---

## How It Works

The simulation consists of a large population of agents controlled entirely by small neural networks initialized with random weights.

Each agent:

* Receives environmental inputs related to:

  * Its own position
  * The approaching obstacle
* Processes these values through a neural network
* Produces movement decisions from the network outputs

Agents survive by navigating through moving wall gaps. When most of the population dies, a new generation is created using only surviving agents as parents.

Children inherit mutated versions of parent neural networks, allowing behaviours to gradually evolve through:

* Selection
* Mutation
* Repeated survival pressure

Over time, stable movement patterns and survival strategies emerge naturally.

---

## Neural Network Visualization

The project includes a live neural visualization panel displaying the current best performing agent’s network in real time.

This allows observation of:

* Input influence
* Hidden layer activation
* Behavioural convergence
* Evolutionary adaptation over generations

The visualization became particularly useful for identifying how environmental variables affected network behaviour.

---

## Emergent Behaviour

One of the most interesting observations during development was how strongly environmental structure influenced evolution. The main factor that i found this in was *Spawn position*, and initaly these gave widly diffrent results but then they all converged across isolated populations.

For example:

* Agents spawned further to the right required significantly longer for stable behaviour to emerge
* Across multiple generations, populations consistently evolved behaviour that moved agents leftward before stabilizing
the Neural visualization showed the `pos x` input strongly influencing `hidden 1` which effects the agent `left` movement, directly tieing their behavior of prioritising the left wall and potentialy the increased "noise" seen when spawed an location further from a wall.

---

## Design Philosophy

This project is intentionally minimal and experimental.

There are:

* No handcrafted rewards
* No explicit punishment systems
* No predefined objectives beyond persistence

Survival itself acts as the only filter.

The project focuses on exploring:

* Emergent behaviour
* Evolution through mutation
* Environmental influence on adaptation
* Stability through persistence

rather than maximizing AI performance or creating a traditional reinforcement learning system.

---

## Future Plans

Planned improvements include:

* Rebuilding the project in React and TypeScript
* Improved rendering and visualization tools
* Cleaner and more interactive UI
* Multiple levels and environmental obstacles
* More advanced evolutionary mechanics
* Better debugging and behavioural analysis systems

---

## Current Features

* Evolutionary neural network agents
* Mutation based inheritance
* Population based selection
* Real time neural network visualization
* Emergent survival behaviour
* Procedurally repeating obstacle environment
* Experimental evolutionary simulation framework
