# MesaUserBehavior

A simulation of users participating in a fixed number of tasks.

## Simulation rules
1. Simulation is initialized with a fixed number of agents and tasks  
1. Each task is represented by a cell in a 2D grid. All tasks have a limit on
   the number of agents it can recruit: `max_agent_per_cell`;
2. At each step each agent decides whether to participate in a task with
      probability, `p_participate`.
3. If the agent is participating, it will joint the first available task
   from its recent participation history within a time window, ordered by highest frequency;
   (To simulate the user behavior of participating in the same task.)
   If no space is available for all these tasks, it will join a random available task.



## To run
1. Install mesa package from https://mesa.readthedocs.io/
2. Run simulation

    cd user_behavior # go to simulation directory
    mkdir output  # make output directory

    python UserModel.py
