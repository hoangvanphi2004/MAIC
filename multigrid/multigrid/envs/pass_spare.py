from __future__ import annotations

from multigrid.base import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.world_object import WorldObj, Door, Wall
from multigrid.core.agent import Agent
from gymnasium import spaces
import numpy as np


class Switch(WorldObj):
    """
    Floor switch that opens the shared door while occupied by any agent.
    """
    type_name = 'switch'

    def __new__(cls):
        # Grey switch tile
        obj = super().__new__(cls, color='grey')
        return obj

    def can_overlap(self) -> bool:
        return True

    def render(self, img):
        from ..utils.rendering import fill_coords, point_in_rect
        # Simple square pad
        fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.9), self.color.rgb())


class PassSparseEnv(MultiGridEnv):
    """
    Pass-sparse: Two agents in two rooms of a 30x30 grid.
    The rooms are separated by a door and agents start in the same (left) room.
    The door opens only when one of the switches is occupied (either room).
    Agents receive a collective positive reward, and the episode terminates
    only when BOTH agents have changed to the other (right) room.

    Observations are the standard partially observable image; the environment
    additionally maintains an internal state vector (x, y for all agents,
    and a binary flag for door open) for potential logging.
    """

    def __init__(
        self,
        size: int = 30,
        num_agents: int = 2,
        max_steps: int | None = None,
        team_reward: float = 1.0,
        only_turn_and_forward: bool = False,
        **kwargs,
    ):
        self.size = size
        self.team_reward = team_reward
        self.only_turn_and_forward = only_turn_and_forward

        # Runtime state
        self.split_x = None
        self.left_switch_pos = None
        self.right_switch_pos = None
        self.door_pos = None
        self.door_obj: Door | None = None

        super().__init__(
            mission_space=(
                'Open the door by standing on a switch and move BOTH agents '
                'to the other room'
            ),
            agents=num_agents,
            grid_size=size,
            max_steps=max_steps or (4 * size * size),
            joint_reward=False,  # we return identical rewards per-agent
            success_termination_mode='all',
            **kwargs,
        )

        if self.only_turn_and_forward:
            for agent in self.agents:
                agent.action_space = spaces.Discrete(2)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Split rooms by a vertical wall; place a single door in that wall
        self.split_x = width // 2
        mid_y = height // 2

        # Build the separating wall except where the door is
        for y in range(1, height - 1):
            if y == mid_y:
                continue
            self.grid.set(self.split_x, y, Wall())

        # Place door (initially closed); we keep a reference to control openness
        self.door_pos = (self.split_x, mid_y)
        door = Door(is_open=False, is_locked=False)
        self.put_obj(door, *self.door_pos)
        self.door_obj = door

        # Place switches: one in each room
        self.left_switch_pos = (self.split_x // 2, mid_y)
        self.right_switch_pos = (width - 2 - (self.split_x // 2), mid_y)
        self.put_obj(Switch(), *self.left_switch_pos)
        self.put_obj(Switch(), *self.right_switch_pos)

        # Place agents in the left room, fixed positions
        start_positions = [
            (2, mid_y - 2),
            (3, mid_y + 2),
        ]
        for i, agent in enumerate(self.agents):
            px, py = start_positions[i % len(start_positions)]
            agent.state.pos = (px, py)
            agent.state.dir = 0  # facing right

    def _update_door_state(self):
        # Door open if ANY agent stands on either switch
        occupied = False
        for st in self.agent_states.pos:
            if tuple(st) == self.left_switch_pos or tuple(st) == self.right_switch_pos:
                occupied = True
                break
        if self.door_obj is not None:
            self.door_obj.is_open = occupied
            # update grid visual/state
            self.grid.update(*self.door_pos)

    def _both_in_right_room(self) -> bool:
        # Right room is x > split_x
        return all(pos[0] > self.split_x for pos in self.agent_states.pos)

    def handle_actions(self, actions):
        rewards = {i: 0.0 for i in range(self.num_agents)}

        # Randomize order
        order = (0,) if self.num_agents == 1 else self.np_random.random(size=self.num_agents).argsort()

        for i in order:
            if i not in actions:
                continue
            agent, action = self.agents[i], actions[i]

            # Map compact action space if enabled
            if self.only_turn_and_forward:
                if action == 0:  # turn left
                    action = 0
                elif action == 1:  # forward
                    action = 2
                else:
                    continue

            if agent.state.terminated:
                continue

            if action == 0:  # left
                agent.state.dir = (agent.state.dir - 1) % 4
            elif action == 1:  # right
                agent.state.dir = (agent.state.dir + 1) % 4
            elif action == 2:  # forward
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)

                if fwd_obj is None or fwd_obj.can_overlap():
                    # prevent agent overlap if disabled
                    if not self.allow_agent_overlap:
                        agent_present = np.bitwise_and.reduce(self.agent_states.pos == fwd_pos, axis=1).any()
                        if agent_present:
                            continue
                    agent.state.pos = fwd_pos
            # ignore other actions

        # After all moves, update door state based on switch occupancy
        self._update_door_state()

        # Success condition: both agents in right room
        if self._both_in_right_room():
            for i in range(self.num_agents):
                rewards[i] += self.team_reward
            # terminate all agents
            self.agent_states.terminated = True

        return rewards

