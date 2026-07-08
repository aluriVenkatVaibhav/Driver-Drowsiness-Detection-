import random
from collections import defaultdict

from game import Agent, Directions
from util import manhattanDistance


class HierarchicalGhostAgent(Agent):
    """
    Hierarchical tabular ghost controller.

    The meta level selects an intent:
      - chase_pacman: close distance when Pacman is vulnerable
      - ambush: move toward Pacman's likely next position when far away
      - guard_capsule: pressure capsules when Pacman can escape through them
      - flee_scared: avoid Pacman while scared

    The low level learns action values over compact local features. A small
    heuristic prior is mixed with learned Q-values so an untrained ghost is
    already sensible, while Q-learning can still improve the policy.
    """

    GOALS = ("chase_pacman", "ambush", "guard_capsule", "flee_scared")

    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.goal_interval = int(kwargs.get("ghost_goal_interval", 8))
        self.epsilon = float(kwargs.get("ghost_epsilon", 0.15))
        self.epsilon_min = float(kwargs.get("ghost_epsilon_end", 0.03))
        self.epsilon_decay = float(kwargs.get("ghost_epsilon_decay", 0.00002))
        self.alpha = float(kwargs.get("ghost_alpha", 0.2))
        self.gamma = float(kwargs.get("ghost_gamma", kwargs.get("gamma", 0.95)))
        self.heuristic_weight = float(kwargs.get("ghost_heuristic_weight", 0.35))
        self.catch_reward = float(kwargs.get("ghost_catch_reward", 250.0))
        self.death_penalty = float(kwargs.get("ghost_eaten_penalty", -250.0))
        self.progress_reward = float(kwargs.get("ghost_progress_reward", 8.0))
        self.scared_distance_reward = float(kwargs.get("ghost_scared_distance_reward", 6.0))
        self.step_penalty = float(kwargs.get("ghost_step_penalty", -0.2))
        self.avoid_stop = kwargs.get("ghost_avoid_stop", True)

        self.q_table = defaultdict(float)
        self.current_goal = "chase_pacman"
        self.goal_step_count = 0
        self.goal_counts = defaultdict(int)
        self.is_learning = False
        self.is_eval = True

    def set_learning(self, is_learning):
        self.is_learning = is_learning
        self.is_eval = not is_learning

    def get_goal_stats(self):
        return dict(self.goal_counts)

    def _legal_actions(self, state):
        legal = state.getLegalActions(self.index)
        if self.avoid_stop and len(legal) > 1 and Directions.STOP in legal:
            legal = [action for action in legal if action != Directions.STOP]
        return legal

    def _ghost_state(self, state):
        return state.getGhostState(self.index)

    def _ghost_pos(self, state):
        ghost_state = self._ghost_state(state)
        if ghost_state.configuration is None:
            return None
        return ghost_state.getPosition()

    def _direction(self, state):
        ghost_state = self._ghost_state(state)
        if ghost_state.configuration is None:
            return Directions.STOP
        return ghost_state.configuration.getDirection()

    def _nearest_capsule_distance(self, state, origin):
        capsules = state.getCapsules()
        if not capsules or origin is None:
            return None
        return min(manhattanDistance(origin, capsule) for capsule in capsules)

    def _target_for_goal(self, state, goal):
        pac_pos = state.getPacmanPosition()
        ghost_pos = self._ghost_pos(state)
        if pac_pos is None or ghost_pos is None:
            return pac_pos

        if goal == "guard_capsule" and state.getCapsules():
            return min(state.getCapsules(), key=lambda cap: manhattanDistance(pac_pos, cap))

        if goal == "ambush":
            direction = state.getPacmanState().configuration.getDirection()
            dx, dy = {
                Directions.NORTH: (0, 1),
                Directions.SOUTH: (0, -1),
                Directions.EAST: (1, 0),
                Directions.WEST: (-1, 0),
            }.get(direction, (0, 0))
            
            walls = state.getWalls()
            cx, cy = int(pac_pos[0]), int(pac_pos[1])
            for _ in range(4):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                    cx, cy = nx, ny
                else:
                    break
            return (cx, cy)

        return pac_pos

    def _select_goal(self, state):
        ghost_state = self._ghost_state(state)
        if ghost_state.configuration is None:
            return "chase_pacman"
        if ghost_state.scaredTimer > 0:
            return "flee_scared"

        ghost_pos = ghost_state.getPosition()
        pac_pos = state.getPacmanPosition()
        dist = manhattanDistance(ghost_pos, pac_pos)
        cap_dist = self._nearest_capsule_distance(state, pac_pos)

        if cap_dist is not None and cap_dist <= 5 and dist > 2:
            return "guard_capsule"
            
        # Multi-ghost coordination: Ghost 2+ prefers to ambush
        if self.index > 1 and dist >= 4:
            return "ambush"

        return "chase_pacman"

    def _sign(self, value):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def _distance_bucket(self, distance):
        return min(int(distance // 2), 8)

    def _features(self, state, goal):
        ghost_state = self._ghost_state(state)
        ghost_pos = self._ghost_pos(state)
        pac_pos = state.getPacmanPosition()
        if ghost_state.configuration is None or ghost_pos is None or pac_pos is None:
            return (goal, 8, 0, 0)

        target = self._target_for_goal(state, goal) or pac_pos
        target_dist = manhattanDistance(ghost_pos, target)
        target_dx = self._sign(target[0] - ghost_pos[0])
        target_dy = self._sign(target[1] - ghost_pos[1])

        return (
            goal,
            self._distance_bucket(target_dist),
            target_dx,
            target_dy,
        )

    def _heuristic_score(self, state, action, goal):
        try:
            successor = state.generateSuccessor(self.index, action)
        except Exception:
            return -1000.0

        ghost_pos = self._ghost_pos(state)
        next_ghost_pos = self._ghost_pos(successor)
        pac_pos = state.getPacmanPosition()
        next_pac_pos = successor.getPacmanPosition()
        if ghost_pos is None or next_ghost_pos is None or pac_pos is None or next_pac_pos is None:
            return -1000.0

        ghost_state = self._ghost_state(state)
        prev_dist = manhattanDistance(ghost_pos, pac_pos)
        next_dist = manhattanDistance(next_ghost_pos, next_pac_pos)

        if successor.isLose():
            return 1000.0

        if ghost_state.scaredTimer > 0 or goal == "flee_scared":
            score = (next_dist - prev_dist) * 6.0
            if next_dist <= 1:
                score -= 80.0
            return score

        target = self._target_for_goal(state, goal) or pac_pos
        prev_target_dist = manhattanDistance(ghost_pos, target)
        next_target_dist = manhattanDistance(next_ghost_pos, target)
        score = (prev_target_dist - next_target_dist) * 6.0

        if next_dist <= 1:
            score += 25.0
        if goal == "guard_capsule" and state.getCapsules():
            score += (prev_target_dist - next_target_dist) * 4.0
        if action == Directions.REVERSE.get(self._direction(state), Directions.STOP):
            score -= 0.5
        return score

    def getAction(self, state):
        legal = self._legal_actions(state)
        if not legal:
            return Directions.STOP

        if self.goal_step_count <= 0 or self.goal_step_count % self.goal_interval == 0:
            self.current_goal = self._select_goal(state)
            self.goal_counts[self.current_goal] += 1
        self.goal_step_count += 1

        if self.is_learning and random.random() < self.epsilon:
            return random.choice(legal)

        features = self._features(state, self.current_goal)
        shuffled = list(legal)
        random.shuffle(shuffled)
        return max(
            shuffled,
            key=lambda action: (
                self.q_table[(features, action)]
                + self.heuristic_weight * self._heuristic_score(state, action, self.current_goal)
            ),
        )

    def _shaped_reward(self, state, action, next_state, base_reward, goal):
        # Ignore base_reward to avoid double-shaping from hierarchical_train.py
        # or score-based noise from train.py.
        reward = self.step_penalty
        ghost_state = self._ghost_state(state)
        next_ghost_state = next_state.getGhostState(self.index)
        ghost_pos = self._ghost_pos(state)
        next_ghost_pos = next_ghost_state.getPosition() if next_ghost_state.configuration else None
        pac_pos = state.getPacmanPosition()
        next_pac_pos = next_state.getPacmanPosition()

        if next_state.isLose():
            reward += self.catch_reward
        if ghost_state.scaredTimer > 0 and next_ghost_state.configuration is None:
            reward += self.death_penalty

        if ghost_pos is None or next_ghost_pos is None or pac_pos is None or next_pac_pos is None:
            return reward

        prev_dist = manhattanDistance(ghost_pos, pac_pos)
        next_dist = manhattanDistance(next_ghost_pos, next_pac_pos)

        if ghost_state.scaredTimer > 0 or goal == "flee_scared":
            reward += (next_dist - prev_dist) * self.scared_distance_reward
            if next_dist <= 1:
                reward += self.death_penalty * 0.25
        else:
            target = self._target_for_goal(state, goal) or pac_pos
            prev_target_dist = manhattanDistance(ghost_pos, target)
            next_target_dist = manhattanDistance(next_ghost_pos, target)
            reward += (prev_target_dist - next_target_dist) * self.progress_reward
            if next_dist <= 1:
                reward += self.progress_reward * 2.0

        return reward

    def update(self, state, action, next_state, reward, done=False, **kwargs):
        if not self.is_learning:
            return None

        goal = self.current_goal
        features = self._features(state, goal)
        next_legal = self._legal_actions(next_state)
        shaped = self._shaped_reward(state, action, next_state, reward, goal)

        if next_legal and not done:
            next_features = self._features(next_state, goal)
            max_next_q = max(
                self.q_table[(next_features, next_action)]
                + self.heuristic_weight * self._heuristic_score(next_state, next_action, goal)
                for next_action in next_legal
            )
        else:
            max_next_q = 0.0

        current_q = self.q_table[(features, action)]
        self.q_table[(features, action)] = current_q + self.alpha * (
            shaped + self.gamma * max_next_q - current_q
        )
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return None

    def final(self, state):
        self.goal_step_count = 0
        self.current_goal = "chase_pacman"

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": dict(self.q_table),
                    "epsilon": self.epsilon,
                    "goal_counts": dict(self.goal_counts),
                },
                f,
            )

    def load(self, path):
        import os
        import pickle
        if os.path.exists(path):
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "q_table" in payload:
                self.q_table = defaultdict(float, payload["q_table"])
                self.epsilon = payload.get("epsilon", self.epsilon)
                self.goal_counts = defaultdict(int, payload.get("goal_counts", {}))
            else:
                self.q_table = defaultdict(float, payload)
