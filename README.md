# Curiousity Driven Reinforcement Learning

## Getting Started

This project uses A3C (Asynchronous Actor Critic) Algorithm along with the ICM (Intrinsic Curiousity Module) to train an agent to follow the red block in the room.

### Tech Stack

* PyTorch 
* gymnasium
* gym-miniworld

## Overview

Some keypoints of this project:

* This project uses multithreading to train multiple local agents in parallel and combining it into a global agent.
* The agent's model is trained for 1 million steps using 12 threads in parallel.
* The model uses CNN architecture to process the agent's environment and derive actions from it.
* The ICM algorithm rewards the agent to explore unexplore states in the environment.


## How training multiple agents in parallel works: 

![a3c](https://github.com/AkshayKulkarni3467/CuriousityDrivenA3C/assets/129979542/af386194-7f49-41ed-9dd1-4670a7c6622d)


## Before training GIF:

![training](https://github.com/AkshayKulkarni3467/CuriousityDrivenA3C/assets/129979542/898ce080-63c1-4670-bd89-37c25aa26283)


## After training GIF:

![evaluation](https://github.com/AkshayKulkarni3467/CuriousityDrivenA3C/assets/129979542/bc06dff6-00d7-4fdd-a94c-b50cf54e2775)



