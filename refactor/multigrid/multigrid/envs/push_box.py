from __future__ import annotations

from gymnasium import spaces
import numpy as np

from multigrid.base import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import DIR_TO_VEC
from multigrid.core.world_object import WorldObj


class PushBoxTile(WorldObj):
    """
    A single tile belonging to the shared 3x3 push box.
    """
    type_name = 'push_box_tile'

    def __new__(cls):
        return super().__new__(cls, color='yellow')

    def can_overlap(self) -> bool:
        return False

    def render(self, img):
        from ..utils.rendering import fill_coords, point_in_rect
        fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), self.color.rgb())
        fill_coords(img, point_in_rect(0.22, 0.78, 0.22, 0.78), (120, 90, 10))


class PushBoxEnv(MultiGridEnv):
    """
    Two-agent cooperative push task.

    A 3x3 box starts near the center of the room. The box moves by one grid cell
    only when at least two agents push it in the same direction during the same
    timestep. The episode succeeds when any part of the box reaches a wall.
    """

    def __init__(
        self,
        size: int = 12,
        num_agents: int = 2,
        max_steps: int | None = None,
        team_reward: float = 100.0,
        only_turn_and_forward: bool = False,
        **kwargs,
    ):
        if size < 9:
            raise ValueError('PushBoxEnv requires size >= 9 for a movable 3x3 box.')
        if num_agents < 2:
            raise ValueError('PushBoxEnv requires at least 2 agents.')

        self.size = size
        self.team_reward = team_reward
        self.only_turn_and_forward = only_turn_and_forward

        self.box_center = None
        self._box_cells: set[tuple[int, int]] = set()
        self._box_reached_wall = False

        super().__init__(
            mission_space='Push the 3x3 box to a wall using two agents',
            agents=num_agents,
            grid_size=size,
            max_steps=max_steps or (4 * size * size),
            joint_reward=False,
            success_termination_mode='all',
            **kwargs,
        )

        if self.only_turn_and_forward:
            for agent in self.agents:
                agent.action_space = spaces.Discrete(2)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.box_center = (width // 2, height // 2)
        self._set_box_at_center(self.box_center)
        self._box_reached_wall = False

        # Place two agents on the left of the box, facing right.
        # Extra agents (if any) are placed nearby in the same lane region.
        start_positions = [
            (self.box_center[0] - 3, self.box_center[1] - 1),
            (self.box_center[0] - 3, self.box_center[1] + 1),
            (self.box_center[0] - 4, self.box_center[1]),
            (self.box_center[0] - 4, self.box_center[1] + 2),
        ]

        for i, agent in enumerate(self.agents):
            px, py = start_positions[i % len(start_positions)]
            px = int(np.clip(px, 1, width - 2))
            py = int(np.clip(py, 1, height - 2))
            # Keep agents outside of the 3x3 box footprint.
            if (px, py) in self._box_cells:
                px = max(1, px - 2)
            agent.state.pos = (px, py)
            agent.state.dir = 0

    def _set_box_at_center(self, center: tuple[int, int]):
        self._box_cells = set()
        cx, cy = center
        for x in range(cx - 1, cx + 2):
            for y in range(cy - 1, cy + 2):
                self.grid.set(x, y, PushBoxTile())
                self._box_cells.add((x, y))

    def _clear_box(self):
        for x, y in self._box_cells:
            self.grid.set(x, y, None)

    def _move_box(self, dx: int, dy: int):
        old_cells = set(self._box_cells)
        self._clear_box()

        self._box_cells = {(x + dx, y + dy) for x, y in old_cells}
        cx, cy = self.box_center
        self.box_center = (cx + dx, cy + dy)

        for x, y in self._box_cells:
            self.grid.set(x, y, PushBoxTile())

    def _box_touches_wall(self) -> bool:
        wall_x_min, wall_x_max = 1, self.width - 2
        wall_y_min, wall_y_max = 1, self.height - 2
        return any(
            x == wall_x_min or x == wall_x_max or y == wall_y_min or y == wall_y_max
            for x, y in self._box_cells
        )

    def _can_move_box(self, dx: int, dy: int) -> bool:
        old_cells = self._box_cells
        new_cells = {(x + dx, y + dy) for x, y in old_cells}

        # Box footprint must stay inside walkable interior (not into perimeter walls).
        if any(x <= 0 or x >= self.width - 1 or y <= 0 or y >= self.height - 1 for x, y in new_cells):
            return False

        # New edge cells must be empty and agent-free.
        entering_cells = new_cells - old_cells
        for pos in entering_cells:
            obj = self.grid.get(*pos)
            if obj is not None and not isinstance(obj, PushBoxTile):
                return False
            if np.bitwise_and.reduce(self.agent_states.pos == pos, axis=1).any():
                return False

        return True

    def _map_action(self, action: int) -> int | None:
        if not self.only_turn_and_forward:
            return action

        if action == 0:
            return 0
        if action == 1:
            return 2
        return None

    def handle_actions(self, actions):
        rewards = {i: 0.0 for i in range(self.num_agents)}

        order = (0,) if self.num_agents == 1 else self.np_random.random(size=self.num_agents).argsort()

        push_intents: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
        mapped_actions = {}

        # First pass: collect mapped actions and push intents.
        for i in order:
            if i not in actions:
                continue

            action = self._map_action(actions[i])
            if action is None:
                continue

            mapped_actions[i] = action
            agent = self.agents[i]
            if agent.state.terminated:
                continue

            if action == 2:
                fwd_pos = agent.front_pos
                if tuple(fwd_pos) in self._box_cells:
                    dvec = DIR_TO_VEC[agent.state.dir]
                    key = (int(dvec[0]), int(dvec[1]))
                    push_intents.setdefault(key, []).append((i, tuple(fwd_pos)))

        successful_pushers: set[int] = set()

        # Resolve a single push direction per step (largest valid pusher group).
        chosen_direction = None
        chosen_pushers = []
        for direction, pushers in sorted(push_intents.items(), key=lambda item: len(item[1]), reverse=True):
            if len(pushers) < 2:
                continue
            dx, dy = direction
            if self._can_move_box(dx, dy):
                chosen_direction = direction
                chosen_pushers = pushers
                break

        if chosen_direction is not None:
            dx, dy = chosen_direction
            self._move_box(dx, dy)
            for i, old_fwd in chosen_pushers:
                # Pusher steps into the vacated cell that was previously box-occupied.
                if not self.allow_agent_overlap:
                    occupied = np.bitwise_and.reduce(self.agent_states.pos == old_fwd, axis=1)
                    occupied[i] = False
                    if occupied.any():
                        continue
                self.agents[i].state.pos = old_fwd
                successful_pushers.add(i)

        # Second pass: handle non-push actions and regular movement.
        for i in order:
            if i not in mapped_actions:
                continue

            agent = self.agents[i]
            if agent.state.terminated:
                continue

            action = mapped_actions[i]

            if action == 0:
                agent.state.dir = (agent.state.dir - 1) % 4
            elif action == 1:
                agent.state.dir = (agent.state.dir + 1) % 4
            elif action == 2:
                if i in successful_pushers:
                    continue

                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)

                # If trying to move into the box without a valid 2-agent push, block movement.
                if tuple(fwd_pos) in self._box_cells:
                    continue

                if fwd_obj is None or fwd_obj.can_overlap():
                    if not self.allow_agent_overlap:
                        agent_present = np.bitwise_and.reduce(self.agent_states.pos == fwd_pos, axis=1).any()
                        if agent_present:
                            continue
                    agent.state.pos = fwd_pos

        # Success: box reaches any wall-adjacent interior line.
        if not self._box_reached_wall and self._box_touches_wall():
            self._box_reached_wall = True
            # Cooperative success: split team reward equally among all agents.
            reward_receivers = set(range(self.num_agents))
            per_agent_reward = self.team_reward / max(1, self.num_agents)
            for i in reward_receivers:
                rewards[i] += per_agent_reward
            self.agent_states.terminated = True

        return rewards
