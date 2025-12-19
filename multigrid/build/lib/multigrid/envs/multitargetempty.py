from __future__ import annotations

from multigrid.base import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Color, Direction, Type
from multigrid.core.world_object import Goal, WorldObj
from multigrid.core.agent import Agent
from typing import SupportsFloat
from gymnasium import spaces
import numpy as np


class AgentGoal(WorldObj):
    """
    Goal object that belongs to a specific agent.
    Only the owner agent can collect this goal.
    """
    
    type_name = 'goal'
    
    def __new__(cls, agent_id: int, color: str = 'green'):
        """
        Parameters
        ----------
        agent_id : int
            The index of the agent that owns this goal
        color : str
            Object color (string like 'red', 'green', etc.)
        """
        obj = super().__new__(cls, color=color)
        obj.agent_id = agent_id
        obj.collected = False
        return obj
    
    def can_overlap(self) -> bool:
        return True
    
    def render(self, img):
        from ..utils.rendering import fill_coords, point_in_rect
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


class MultiTargetEmptyEnv(MultiGridEnv):
    """
    Multi-agent environment where each agent has their own target goal.
    
    ***********
    Description
    ***********
    
    This environment is an empty room where each agent has their own colored goal.
    Any agent can collect any goal and, upon collection, that goal disappears
    from the grid. The collecting agent receives an individual reward. Once all
    goals are collected, all agents receive a large team bonus reward.
    
    Features:
    - All goals are yellow colored
    - Any agent can collect any goal
    - Collected goals disappear from the grid
    - Individual reward for collecting a goal
    - Team bonus reward when all goals are collected
    - Small step penalty when no reward is received in a timestep
    
    *************
    Mission Space
    *************
    
    "collect all goals together as a team"
    
    *****************
    Observation Space
    *****************
    
    The multi-agent observation space is a Dict mapping from agent index to
    corresponding agent observation space.
    
    Each agent observation is a dictionary with the following entries:
    
    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's partially observable view of the environment
    * direction : int
        Agent's direction (0: right, 1: down, 2: left, 3: up)
    * mission : Mission
        Task string corresponding to the current environment configuration
    
    ************
    Action Space
    ************
    
    The multi-agent action space is a Dict mapping from agent index to
    corresponding agent action space.
    
    Agent actions are discrete integer values, given by:
    
    +-----+--------------+-----------------------------+
    | Num | Name         | Action                      |
    +=====+==============+=============================+
    | 0   | left         | Turn left                   |
    +-----+--------------+-----------------------------+
    | 1   | right        | Turn right                  |
    +-----+--------------+-----------------------------+
    | 2   | forward      | Move forward                |
    +-----+--------------+-----------------------------+
    | 3   | pickup       | Pick up an object           |
    +-----+--------------+-----------------------------+
    | 4   | drop         | Drop an object              |
    +-----+--------------+-----------------------------+
    | 5   | toggle       | Toggle / activate an object |
    +-----+--------------+-----------------------------+
    | 6   | done         | Done completing task        |
    +-----+--------------+-----------------------------+
    
    *******
    Rewards
    *******
    
    - Individual reward: ``individual_reward`` when agent reaches their goal
    - Team bonus: ``team_bonus_reward`` when all agents reach their goals
    - Step penalty: ``-step_penalty`` each timestep an agent receives no reward
    
    ***********
    Termination
    ***********
    
    The episode ends when:
    
    * All agents have reached their goals
    * Timeout (see ``max_steps``)
    
    *************************
    Registered Configurations
    *************************
    
    * ``MultiGrid-MultiTargetEmpty-8x8-v0``
    * ``MultiGrid-MultiTargetEmpty-16x16-v0``
    """
    
    def __init__(
        self,
        size: int = 8,
        num_agents: int = 2,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: Direction | None = None,
        max_steps: int | None = None,
        individual_reward: float = 1.0,
        team_bonus_reward: float = 5.0,
        step_penalty: float = 0.1,
        only_turn_and_forward: bool = False,
        **kwargs):
        """
        Parameters
        ----------
        size : int, default=8
            Width and height of the grid
        num_agents : int, default=2
            Number of agents in the environment
        agent_start_pos : tuple[int, int], optional
            Starting position of all agents (random if None)
        agent_start_dir : Direction, optional
            Starting direction of all agents (random if None)
        max_steps : int, optional
            Maximum number of steps per episode
        individual_reward : float, default=1.0
            Reward given to an agent when they reach their own goal
        team_bonus_reward : float, default=5.0
            Bonus reward given to all agents when all goals are collected
        step_penalty : float, default=0.01
            Small penalty subtracted each timestep an agent receives no reward
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.individual_reward = individual_reward
        self.team_bonus_reward = team_bonus_reward
        self.step_penalty = step_penalty
        # If True, limit the per-agent action space to two actions:
        # 0 -> turn (left)
        # 1 -> move forward
        # Note: turning right is not available in this mode (you can turn
        # left multiple times to rotate right).
        self.only_turn_and_forward = only_turn_and_forward
        
        # Track which agents have collected their goals
        self.goals_collected = None
        self.agent_goals = {}  # Maps agent_id to their goal object
        self.remaining_goals = 0  # Count of goals left on the grid
        
        super().__init__(
            mission_space="collect all goals together as a team",
            agents=num_agents,
            grid_size=size,
            max_steps=max_steps or (4 * size**2),
            joint_reward=False,  # Individual rewards per agent
            success_termination_mode='all',  # Terminate when all agents reach goals
            **kwargs,
        )

        # If requested, override each agent's action space to only include
        # two discrete actions: `turn` and `forward`.
        if self.only_turn_and_forward:
            for agent in self.agents:
                agent.action_space = spaces.Discrete(2)
    
    def _gen_grid(self, width, height):
        """
        Generate the grid for a new episode.
        """
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Reset goal collection tracking
        self.goals_collected = [False] * self.num_agents
        self.agent_goals = {}
        self.remaining_goals = self.num_agents
        
        # Color palettes
        agent_color_list = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
        goal_color_list = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
        
        # Place agent-specific goals at different positions
        goal_positions = self._generate_goal_positions(width, height, self.num_agents)
        
        for i, agent in enumerate(self.agents):
            # Agent appearance color (unique per agent)
            agent_color_str = agent_color_list[i % len(agent_color_list)]
            # All goals use yellow color
            goal_color_str = 'yellow'

            # Create a goal for this agent
            goal = AgentGoal(agent_id=i, color=goal_color_str)
            
            # Place the goal
            pos = goal_positions[i]
            self.put_obj(goal, pos[0], pos[1])
            self.agent_goals[i] = goal
            
            # Set agent color to a unique color (decoupled from goal color)
            agent.color = agent_color_str
        
        # Place the agents
        for agent in self.agents:
            if self.agent_start_pos is not None and self.agent_start_dir is not None:
                agent.state.pos = self.agent_start_pos
                agent.state.dir = self.agent_start_dir
            elif self.num_agents == 1:
                # For single agent, use fixed starting position
                agent.state.pos = (1, 1)
                agent.state.dir = 0
            else:
                self.place_agent(agent)
    
    def _generate_goal_positions(self, width, height, num_agents):
        """
        Generate positions for goals, trying to spread them out.
        """
        positions = []
        
        if num_agents == 1:
            # Place goal at a fixed position for single agent
            positions = [(width - 2, height - 2)]
        elif num_agents == 2:
            # Place goals at opposite corners
            positions = [(width - 2, height - 2), (1, 1)]
        elif num_agents == 3:
            # Place goals at three corners/edges
            positions = [(width - 2, height - 2), (1, 1), (width - 2, 1)]
        elif num_agents == 4:
            # Place goals at all four corners
            positions = [(width - 2, height - 2), (1, 1), (width - 2, 1), (1, height - 2)]
        else:
            # For more agents, distribute randomly
            for _ in range(num_agents):
                pos = self.place_obj(None, reject_fn=lambda _, p: p in positions)
                positions.append(pos)
        
        return positions[:num_agents]
    
    def handle_actions(self, actions):
        """
        Override to handle goal collection: any agent can collect any goal.
        """
        rewards = {agent_index: 0 for agent_index in range(self.num_agents)}
        got_reward_step = [False] * self.num_agents
        
        # Randomize agent action order
        if self.num_agents == 1:
            order = (0,)
        else:
            order = self.np_random.random(size=self.num_agents).argsort()
        
        # Track if all goals were just completed this step
        all_collected_before = all(self.goals_collected)
        
        # Update agent states and handle goal collection
        for i in order:
            if i not in actions:
                continue
            
            agent, action = self.agents[i], actions[i]

            # If the environment was configured to expose only two actions,
            # map the incoming compact action to the internal action codes:
            # compact 0 -> turn left (internal 0)
            # compact 1 -> forward   (internal 2)
            if self.only_turn_and_forward:
                if action == 0:
                    mapped_action = 0
                elif action == 1:
                    mapped_action = 2
                else:
                    # Unknown action in the compact space; ignore
                    continue

                action = mapped_action
            
            if agent.state.terminated:
                continue
            
            # Rotate left
            if action == 0:  # Action.left
                agent.state.dir = (agent.state.dir - 1) % 4
            
            # Rotate right
            elif action == 1:  # Action.right
                agent.state.dir = (agent.state.dir + 1) % 4
            
            # Move forward
            elif action == 2:  # Action.forward
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)
                
                if fwd_obj is None or fwd_obj.can_overlap():
                    # Check for agent overlap if not allowed
                    if not self.allow_agent_overlap:
                        agent_present = np.bitwise_and.reduce(
                            self.agent_states.pos == fwd_pos, axis=1).any()
                        if agent_present:
                            continue
                    
                    agent.state.pos = fwd_pos
                    
                    # Check if agent reached a goal
                    if fwd_obj is not None and fwd_obj.type == Type.goal:
                        if getattr(fwd_obj, 'collected', False):
                            continue

                        owner_id = getattr(fwd_obj, 'agent_id', None)
                        # Mark collected flags
                        if owner_id is not None and not self.goals_collected[owner_id]:
                            self.goals_collected[owner_id] = True
                        fwd_obj.collected = True
                        self.remaining_goals = max(0, self.remaining_goals - 1)

                        # Remove the goal from the grid upon collection
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)

                        # Give individual reward with optional decay to the collecting agent
                        reward = self._calculate_reward(self.individual_reward)
                        rewards[i] += reward
                        got_reward_step[i] = True
                        #print(f"agent {i} collected goal of agent {owner_id}, reward: {reward:.2f}, step_count: {self.step_count}, max_steps: {self.max_steps}")

                        # Check if all goals are now collected
                        if self.remaining_goals == 0:
                            # Give team bonus to all agents
                            team_bonus = self._calculate_reward(self.team_bonus_reward)
                            for agent_idx in range(self.num_agents):
                                rewards[agent_idx] += team_bonus
                                got_reward_step[agent_idx] = True
                            #print(f"team bonus reward {team_bonus} granted to all agents!, step_count: {self.step_count}, max_steps: {self.max_steps}")


                            # Terminate all agents
                            self.agent_states.terminated = True
            
            # Other actions (pickup, drop, toggle, done) - not used in this environment
            elif action in [3, 4, 5, 6]:
                pass
        
        # Apply step penalty to agents that received no reward this step
        for idx in range(self.num_agents):
            if not got_reward_step[idx]:
                rewards[idx] -= self.step_penalty
        
        return rewards
    
    def _calculate_reward(self, base_reward):
        """
        Calculate reward (no decay; decay replaced by step penalty).
        """
        return base_reward
    
    def _reward(self):
        """
        Override default reward computation (not used in this environment).
        """
        return 0.0
