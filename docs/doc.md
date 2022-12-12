# Documention of the System

The project is to minise the cost of production of energy in smart grids. We follow RL to optimize the cost. 
The system is fed by inputs and it outputs the decisions that can be implemented on a smart grids. The system
contains following components.
* Wind power plant
* Solar power plant 
* Battery storage
* Diesel Generator

## The System
### Weather forcasting
**We don't do weather forcasting**. For the research we can put some random data on there. 
(future: In real time system we can use some API to get the data)

## Unit time
**todo**

## Inputs
* Weather forcasting data (a probability distribution)
* Current state of the system 

## Outputs
* The probability distribution on set of actions. These actions can be use the solar power plant, turn off 
the diesel generator etc..
