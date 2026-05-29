import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional


import gymnasium as gym
import multigrid.envs
from multigrid.envs.findgoal import FindGoalEnv
from multigrid.core.actions import NavigationAction
from multigrid.core.grid import Type

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional

# ── A* Core ───────────────────────────────────────────────────────────────────

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(pos: Tuple[int, int], env) -> List[Tuple[int, int]]:
    """
    Return walkable neighbors of a grid position.
    A cell is walkable if it's empty or is the goal.
    """
    x, y = pos
    neighbors = []

    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # up, down, left, right
        nx, ny = x + dx, y + dy

        # Bounds check
        if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height:
            continue

        cell = env.grid.get(nx, ny)

        # Walkable if empty or goal
        if cell is None or cell.type == Type.goal:
            neighbors.append((nx, ny))

    return neighbors


def astar(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    env) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding from start to goal on the env grid.

    Returns
    -------
    path : list of (x, y) positions from start to goal (inclusive),
           or None if no path found.
    """
    # Priority queue: (f_score, counter, position)
    # counter breaks ties deterministically
    counter = 0
    open_set = [(0, counter, start)]
    
    came_from: Dict[Tuple, Tuple] = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, env):
            tentative_g = g_score[current] + 1  # uniform cost (each step = 1)

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

    return None  # no path found


# ── Path → Actions ────────────────────────────────────────────────────────────

# Direction vectors matching the env convention
# dir: 0=right, 1=down, 2=left, 3=up
DIR_TO_VEC = {
    0: (1,  0),   # right
    1: (0,  1),   # down
    2: (-1, 0),   # left
    3: (0, -1),   # up
}

VEC_TO_DIR = {v: k for k, v in DIR_TO_VEC.items()}


def path_to_actions(
    path: List[Tuple[int, int]],
    start_dir: int) -> List[NavigationAction]:
    """
    Convert a list of grid positions into a sequence of NavigationActions.

    An agent must:
      1. Rotate (left/right) until facing the next cell
      2. Move forward

    Parameters
    ----------
    path : list of (x, y) — from start to goal inclusive
    start_dir : int — agent's initial direction (0=right,1=down,2=left,3=up)

    Returns
    -------
    actions : list of NavigationAction
    """
    actions = []
    current_dir = start_dir

    for i in range(len(path) - 1):
        cx, cy = path[i]
        nx, ny = path[i + 1]

        # Desired direction vector
        desired_vec = (nx - cx, ny - cy)
        desired_dir = VEC_TO_DIR[desired_vec]

        # Rotate until facing the desired direction
        while current_dir != desired_dir:
            # Choose shortest rotation: left or right
            left_dir  = (current_dir - 1) % 4
            right_dir = (current_dir + 1) % 4

            # Count steps for each option
            steps_left  = (current_dir - desired_dir) % 4
            steps_right = (desired_dir - current_dir) % 4

            if steps_right <= steps_left:
                actions.append(NavigationAction.right)
                current_dir = right_dir
            else:
                actions.append(NavigationAction.left)
                current_dir = left_dir

        # Move forward
        actions.append(NavigationAction.forward)

    return actions



# ── Multi-Agent A* Controller ─────────────────────────────────────────────────

class AStarController:
    """
    Runs A* for each agent and executes their action sequences step by step.
    Replans if an agent gets stuck (e.g. blocked by another agent).
    """

    def __init__(self, env: FindGoalEnv):
        self.env = env
        self.action_queues: Dict[int, List[NavigationAction]] = {}
        self.paths: Dict[int, List[Tuple[int, int]]] = {}

    def plan(self, goal_pos: Tuple[int, int]):
        """Compute A* paths for all agents toward goal_pos."""
        self.action_queues = {}
        self.paths = {}

        for agent in self.env.agents:
            start = tuple(agent.state.pos)
            goal  = tuple(goal_pos)

            path = astar(start, goal, self.env)

            if path is None:
                print(f"Agent {agent.index}: No path found from {start} to {goal}!")
                self.action_queues[agent.index] = []
                continue

            actions = path_to_actions(path, int(agent.state.dir))
            self.action_queues[agent.index] = actions
            self.paths[agent.index] = path
            print(f"Agent {agent.index}: path length={len(path)}, "
                  f"actions={len(actions)}, {start} → {goal}")

    def get_actions(self) -> Dict[int, NavigationAction]:
        """
        Pop the next action for each agent.
        If an agent's queue is empty, it sends 'done'.
        """
        actions = {}
        for agent in self.env.agents:
            idx = agent.index
            if self.action_queues.get(idx):
                actions[idx] = self.action_queues[idx].pop(0)
            else:
                actions[idx] = NavigationAction.done
        return actions

    def replan(self, goal_pos: Tuple[int, int]):
        """Replan from current agent positions (call if stuck)."""
        print("Replanning...")
        self.plan(goal_pos)


# ── Run the full episode ───────────────────────────────────────────────────────

def run_astar_episode(env: FindGoalEnv, render: bool = True, max_replan: int = 3):
    """
    Run a full episode using A* to navigate all agents to the goal.

    Parameters
    ----------
    env : FindGoalEnv
    render : bool — whether to render each step
    max_replan : int — how many times to replan if agents get stuck
    """
    obs, info = env.reset()
    goal_pos = tuple(env.goal_pos)
    print(f"Goal at: {goal_pos}")

    controller = AStarController(env)
    controller.plan(goal_pos)

    step = 0
    replan_count = 0
    prev_positions = {i: None for i in range(env.num_agents)}
    stuck_counter  = {i: 0    for i in range(env.num_agents)}
    STUCK_THRESH = 5  # replan after this many steps without moving

    while not env.is_done():
        actions = controller.get_actions()
        obs, rewards, terminations, truncations, info = env.step(actions)
        step += 1

        if render:
            env.render()

        # Detect stuck agents and replan
        for agent in env.agents:
            current_pos = tuple(agent.state.pos)
            if current_pos == prev_positions[agent.index]:
                stuck_counter[agent.index] += 1
            else:
                stuck_counter[agent.index] = 0
            prev_positions[agent.index] = current_pos

        if any(v >= STUCK_THRESH for v in stuck_counter.values()):
            if replan_count < max_replan:
                replan_count += 1
                controller.replan(goal_pos)
                stuck_counter = {i: 0 for i in range(env.num_agents)}
            else:
                print("Max replans reached, stopping.")
                break

        # Check termination
        if all(terminations.values()) or all(truncations.values()):
            break

    print(f"Episode finished in {step} steps. "
          f"Terminated: {terminations}, Truncated: {truncations}")
    return step

