"""
Mods:
1. Health regeneration (very slow)
2. Bullet damage / 2 for player health
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.envs.registration import register
from engine import GameEngine
from controller import GameController
from settings import SCREEN_WIDTH, SCREEN_HEIGHT

pygame.init()
pygame.mixer.init()

# Not required unless we want to provide traditional gym.make capabilities
register(id='Sidescroller-v0',
         entry_point='MH_env:ShooterEnv',
         max_episode_steps=5000)


class ShooterEnv(gym.Env):
    '''
    Wrapper class that creates a gym interface to the original game engine.
    '''

    # Hints for registered environments; ignored otherwise
    metadata = {
        'render_modes': ['human'],
        'render_fps': 60
    }

    def __init__(self, render_mode=None):
        '''
        Initializes a new Gymnasium environment for the Shooter Game. Loads the
        game engine into the background and defines the action space and the
        observation space.
        '''

        super().__init__()
        self.render_mode = render_mode
        pygame.display.init()
        if self.render_mode == 'human':
            pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Shooter')
            self.screen = pygame.display.get_surface()
            self.game = GameEngine(self.screen, False)
        else:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
            self.game = GameEngine(None, False)

        self.controller = GameController()

        # Discrete action space: 7 possible moves
        self.action_space = Discrete(7)

        # Observation: [dx, dy, health, exit_dx, exit_dy, ammo, grenades]
        low = np.array([-10000, -1000, 0, -10000, -10000, 0, 0], dtype=np.float32)
        high = np.array([10000, 1000, 100, 10000, 10000, 50, 20], dtype=np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)


    def reset(self, seed=None, options=None):
        '''
        Resets the game environment for the beginning of another episode.
        '''
        self.step_count = 0
        self.game.reset_world()
        self.game.load_current_level()

        # Tracks observation and reward values across steps
        self.start_x = self.game.player.rect.centerx
        self.start_y = self.game.player.rect.centery

        # Initialize the variables I decided were important for debugging
        debug_info = {
            'player_health': self.game.player.health,
            'player_distance': (0, 0),
            'exit_distance': self._get_exit_offset(self.game.player)
        }

        # Initialize the previous distance to the goal
        exit_dx, exit_dy = self._get_exit_offset(self.game.player)
        self.prev_goal_distance = abs(exit_dx) + abs(exit_dy)  # Manhattan distance

        # Allow left movement only if the agent passes the goal
        self.allow_left = False

        # Return the initial game state
        observation, debug_info = self._get_observation()
        return observation, debug_info


    def step(self, action):
        '''
        Agent performs a single action in the game environment.
        '''
        controller = self._action_to_controller(action)
        self.game.update(controller)
        self.step_count += 1

        observation, debug_info = self._get_observation()
        reward = self._get_reward()
        terminated = not self.game.player.alive or self.game.level_complete
        truncated = self.step_count >= 2000  # originally 1000

        # Reset the allow_left flag when the next level starts
        if self.game.level_complete:
            self.allow_left = False

        return observation, reward, terminated, truncated, debug_info


    def render(self):
        ''' 
        Visually renders the game so that viewers can watch the agent play. The
        first time this function is called, it initializes the PyGame display
        and mixer (just like a real game). If the self. Every time that it is called, this
        function draws the game.
        '''
        if self.render_mode == 'human':
            # Ensure the game engine's screen is updated
            self.game.screen.fill((0, 0, 0))  # Clear the screen with a black background
            self.game.draw()  # Call the game engine's draw method
            pygame.display.flip()  # Update the display

    def _get_observation(self):
        """ Gets the current observation of the game environment """
        p = self.game.player

        # Distance from start
        p_dx = p.rect.centerx - self.start_x
        p_dy = p.rect.centery - self.start_y

        # Exit distance
        exit_dx, exit_dy = self._get_exit_offset(p)

        # Create an observation (7 values)
        obs = [
            p_dx,
            p_dy,
            p.health,
            exit_dx,
            exit_dy,
            p.ammo,
            p.grenades
        ]

        # Create debug information
        debug_info = {
            'player_health': p.health,
            'player_distance': (p_dx, p_dy),
            'exit_distance': (exit_dx, exit_dy),
        }

        return np.array(obs, dtype=np.float32), debug_info

    # AI helped a lot with these calculations
    def _get_exit_offset(self, player):
        """ Calculate the distance to the exit tile """

        min_dist = float('inf')
        closest_dx, closest_dy = 9999, 9999

        for tile in self.game.groups['exit']:
            dx = tile.rect.centerx - player.rect.centerx
            dy = tile.rect.centery - player.rect.centery
            dist = abs(dx) + abs(dy)
            if dist < min_dist:
                min_dist = dist
                closest_dx = dx
                closest_dy = dy

        return closest_dx, closest_dy

    def _get_reward(self):
        """ Calculate the reward for current step """
        
        # Base reward components
        reward = 0
        player = self.game.player

        # Get current ammo
        ammo = player.ammo

        # Check if player died
        if not self.game.player.alive:
            return -100  # Large penalty for dying

        # Check if level completed
        if self.game.level_complete:
            reward += 200  # Large reward for completing the level

        # Health-based reward
        health_reward = player.health / 100  # Normalize health to 0-1 range
        reward += health_reward * 0.1  # Small constant bonus for higher health

        # Horizontal movement rewards
        forward_movement = player.rect.centerx - self.start_x
        reward += 5 * forward_movement  # Large reward for moving forward

        # Velocity-based movement quality rewards
        if player.vel_x > 0:  # Moving right
            reward += 2 * player.vel_x  # Bonus for faster right movement
        # Even though the agent can only move left after the goal,
        # I still want to penalize it for overshooting
        elif player.vel_x < 0:  # Moving left
            reward -= 0.1 * abs(player.vel_x)  # Small penalty for moving left

        # Jumping reward
        if player.vel_y < 0:  # Negative vel_y means moving upward
            reward += 2  # Large reward for upward jump motion
        elif player.vel_y > 0:  # Positive vel_y means falling
            reward += 0.1  # Tiny reward for being in air

        # Penalty for shooting
        if player.ammo < ammo:
            ammo_penalty = 0.5 / max(player.ammo + 1, 1)  # Lower penalty for shooting
            reward -= ammo_penalty

        # Goal distance reward - encouraged to move to finish
        exit_dx, exit_dy = self._get_exit_offset(player)
        current_goal_distance = abs(exit_dx) + abs(exit_dy)  # Manhattan distance
        distance_delta = self.prev_goal_distance - current_goal_distance

        if distance_delta > 0:
            reward += 10 * distance_delta  # Reward for getting closer to the goal
        elif distance_delta < 0:
            reward -= 5 * abs(distance_delta)  # Penalty for moving away from the goal

        # Enable walking left if the agent has passed the goal
        if exit_dx + 100 < 0:  # Agent has passed the goal + 100 for insurance
            self.allow_left = True

        # Update the previous goal distance
        self.prev_goal_distance = current_goal_distance

        # Penalize for proximity to water - makes agent jump away from water
        closest_water_distance = self._get_closest_water_distance(player)
        if closest_water_distance < 100:  # Apply penalty if within 100 pixels of water
            reward -= (100 - closest_water_distance) * 0.1  # Larger penalty for being closer

        # Survival reward
        reward += 0.05  # Tiny reward per frame for surviving

        return reward
    
    # AI helped a lot with these calculations
    def _get_closest_water_distance(self, player):
        """ Calculate distance to water tiles """
        
        min_dist = float('inf')

        for tile in self.game.groups['water']:
            dx = tile.rect.centerx - player.rect.centerx
            dy = tile.rect.centery - player.rect.centery
            dist = abs(dx) + abs(dy)  # Manhattan distance
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def _action_to_controller(self, action):
        '''
        Converts an action (just an integer) to a game controller object.
        '''
        ctrl = GameController()
        player = self.game.player

        # Detect if the agent is near the edge of the ground - AI
        is_near_edge = not any(
            tile.rect.collidepoint(player.rect.midbottom[0] + 5, player.rect.midbottom[1] + 1)
            for tile in self.game.groups['obstacle']
        )

        # If the agent is near the edge and moving right, force it to jump - no walking off ledges!
        if is_near_edge and action == 1:
            ctrl.jump = ctrl.mright = True
            return ctrl

        # If the agent has passed the goal, force it to walk left and ignore all other actions
        if self.allow_left:
            ctrl.mleft = True
            return ctrl

        # Normal action mapping - took away throwing grenades because of the RNG that comes with them
        elif action == 1:  # Walk right
            ctrl.mright = True
        elif action == 2:  # Jump
            ctrl.jump = True
        elif action == 4:  # Jump + Walk right
            ctrl.jump = ctrl.mright = True
        elif action == 5:  # Shoot
            ctrl.shoot = True

        return ctrl