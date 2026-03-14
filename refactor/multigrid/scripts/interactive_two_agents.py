import argparse
import json

import gymnasium as gym
import multigrid.envs
import pygame

from multigrid.core.actions import Action


# Direction encoding in MultiGrid: 0=right, 1=down, 2=left, 3=up
ABS_DIR_FROM_KEY_ASDW = {
    pygame.K_d: 0,
    pygame.K_s: 1,
    pygame.K_a: 2,
    pygame.K_w: 3,
}

ABS_DIR_FROM_KEY_ARROWS = {
    pygame.K_RIGHT: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_UP: 3,
}


def action_toward_absolute_direction(current_dir: int, target_dir: int) -> Action:
    """
    Convert an absolute direction request into one discrete MultiGrid action.

    Because MultiGrid uses relative controls (turn left/right, then forward), this
    helper returns the single next action required to move toward target_dir.
    """
    delta = (target_dir - current_dir) % 4
    if delta == 0:
        return Action.forward
    if delta == 1:
        return Action.right
    if delta == 3:
        return Action.left
    # Opposite direction: rotate one step now, user can press again next step.
    return Action.right


def key_to_agent_action(key: int, agent_direction: int, agent_id: int) -> Action | None:
    """
    Map keyboard input to one agent action.

    Agent 0: ASDW keys
    Agent 1: Arrow keys
    """
    if agent_id == 0 and key in ABS_DIR_FROM_KEY_ASDW:
        return action_toward_absolute_direction(
            current_dir=agent_direction,
            target_dir=ABS_DIR_FROM_KEY_ASDW[key],
        )

    if agent_id == 1 and key in ABS_DIR_FROM_KEY_ARROWS:
        return action_toward_absolute_direction(
            current_dir=agent_direction,
            target_dir=ABS_DIR_FROM_KEY_ARROWS[key],
        )

    return None


def print_controls() -> None:
    print("\nInteractive 2-agent controls")
    print("- Agent 0: W/A/S/D")
    print("- Agent 1: Arrow Keys")
    print("- R: Reset episode")
    print("- Q or ESC: Quit\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="MultiGrid-Empty-8x8-v0",
        help="MultiGrid environment id",
    )
    parser.add_argument(
        "--env-config",
        type=json.loads,
        default={},
        help="Environment config as JSON string, e.g. '{\"size\": 8}'",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fps", type=int, default=30, help="Render/update FPS")
    args = parser.parse_args()

    env_config = dict(args.env_config)
    env_config.update(render_mode="human")

    # Some registered envs (e.g. PushBox) already provide num_agents in kwargs.
    # Passing agents again can cause: multiple values for keyword argument 'agents'.
    spec = gym.spec(args.env)
    registered_kwargs = dict(spec.kwargs) if spec is not None and spec.kwargs else {}
    has_registered_agent_count = (
        "num_agents" in registered_kwargs or "agents" in registered_kwargs
    )
    has_user_agent_count = "num_agents" in env_config or "agents" in env_config

    make_kwargs = dict(env_config)
    if not has_registered_agent_count and not has_user_agent_count:
        make_kwargs["agents"] = 2

    env = gym.make(args.env, **make_kwargs)

    observations, _ = env.reset(seed=args.seed)
    print_controls()

    running = True
    clock = pygame.time.Clock()

    try:
        while running:
            actions = {0: Action.done, 1: Action.done}
            did_step = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                    break

                if event.key == pygame.K_r:
                    observations, _ = env.reset(seed=args.seed)
                    print("Episode reset")
                    continue

                action_0 = key_to_agent_action(
                    key=event.key,
                    agent_direction=observations[0]["direction"],
                    agent_id=0,
                )
                action_1 = key_to_agent_action(
                    key=event.key,
                    agent_direction=observations[1]["direction"],
                    agent_id=1,
                )

                if action_0 is not None:
                    actions[0] = action_0
                    did_step = True
                if action_1 is not None:
                    actions[1] = action_1
                    did_step = True

            if not running:
                break

            if did_step:
                observations, rewards, terminations, truncations, _ = env.step(actions)
                if any(float(r) != 0.0 for r in rewards.values()):
                    print(f"Rewards: {dict(rewards)}")

                if any(terminations.values()) or any(truncations.values()):
                    print("Episode ended. Auto reset.")
                    observations, _ = env.reset(seed=args.seed)
            else:
                env.render()

            clock.tick(args.fps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
