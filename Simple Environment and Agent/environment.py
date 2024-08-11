class Environment:
    def __init__(self):
        # Initialize the number of timestep that an agent can interact with environment.
        self.limit_of_steps = 10

        # Size of the grid
        self.dim_x = 1
        self.dim_y = 1

        # Starting location of agent
        self.x = 0
        self.y = 0

        # Location of the reward
        self.reward_loc_x = 1
        self.reward_loc_y = 1

    # Return the location of the agent (Current environment's observation)
    def get_location(self) -> list[int]:
        return [self.x, self.y]

    # Update the location of agent based on the action
    def update_location(self, action: str):
        if action == 'UP':
            if self.x != 0:
                self.x -= 1
        elif action == 'DOWN':
            if self.x != self.dim_x:
                self.x += 1
        elif action == 'LEFT':
            if self.y != 0:
                self.y -= 1
        elif action == 'RIGHT':
            if self.y != self.dim_y:
                self.y += 1

    # Check if the agent is in the reward location
    def check_reward(self):
        if self.x == self.reward_loc_x and self.y == self.reward_loc_y:
            return 1
        else:
            return 0

    # The set of allowed actions in current state
    @staticmethod
    def get_actions():
        return ['UP', 'LEFT', 'DOWN', 'RIGHT']

    # Signals the end of an episode
    def is_done(self) -> bool:
        return self.limit_of_steps == 0

    # Handle an agent's action and return reward
    def action(self, action: str) -> float:
        if self.is_done():
            raise Exception('Episode is over.')
        self.limit_of_steps -= 1
        self.update_location(action)
        reward = self.check_reward()
        return reward
