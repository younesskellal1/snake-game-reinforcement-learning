import pygame
import numpy as np
import random
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import style
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from datetime import datetime

style.use("ggplot")

class SnakeGameEnv:
    def __init__(self, width=20, height=20, block_size=20, speed=5):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.display_width = width * block_size
        self.display_height = height * block_size
        self.speed = speed
        
        # Initialize pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Snake - Deep Reinforcement Learning')
        self.clock = pygame.time.Clock()
        
        # Enhanced color palette
        self.white = (255, 255, 255)
        self.black = (15, 15, 25)
        self.dark_gray = (30, 30, 40)
        self.light_gray = (60, 60, 70)
        self.red = (255, 80, 80)
        self.green = (50, 255, 100)
        self.blue = (80, 150, 255)
        self.yellow = (255, 255, 100)
        self.purple = (200, 100, 255)
        self.orange = (255, 150, 50)
        
        # Game state
        self.snake_pos = None
        self.food_pos = None
        self.score = None
        self.snake_dir = None
        self.game_over = None
        self.steps_without_food = None
        self.max_steps_without_food = width * height * 2  # More lenient timeout
        self.total_steps = 0
        
        # Enhanced fonts
        try:
            self.title_font = pygame.font.Font(None, 35)
            self.score_font = pygame.font.Font(None, 28)
            self.info_font = pygame.font.Font(None, 22)
            self.small_font = pygame.font.Font(None, 18)
        except:
            # Fallback to system font
            self.title_font = pygame.font.SysFont(None, 35)
            self.score_font = pygame.font.SysFont(None, 28)
            self.info_font = pygame.font.SysFont(None, 22)
            self.small_font = pygame.font.SysFont(None, 18)
        
        # UI state
        self.paused = False
        self.show_help = False
        
        # Animation variables
        self.food_pulse = 0
        self.snake_glow = 0
        
        # Particle effects
        self.particles = []
        self.food_eaten_particles = []
        
        # Score popup effects
        self.score_popups = []
        
        # Initialize game
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        # Random starting position and direction
        start_x = random.randint(2, self.width - 3)
        start_y = random.randint(2, self.height - 3)
        self.snake_pos = [(start_x, start_y)]
        
        # Random starting direction
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.snake_dir = random.choice(directions)
        
        self.place_food()
        self.score = 0
        self.game_over = False
        self.steps_without_food = 0
        self.total_steps = 0
        
        # Clear visual effects
        self.particles.clear()
        self.food_eaten_particles.clear()
        self.score_popups.clear()
        
        return self._get_state()
    
    def place_food(self):
        """Place food in a random position not occupied by the snake"""
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake_pos:
                self.food_pos = (x, y)
                break
            attempts += 1
        
        # If we can't place food (snake fills board), game over
        if attempts >= 100:
            self.game_over = True
    
    def _get_state(self):
        """Enhanced state representation with more spatial awareness"""
        head_x, head_y = self.snake_pos[0]
        food_x, food_y = self.food_pos
        
        # Direction encoding (one-hot)
        dir_up = int(self.snake_dir == (0, -1))
        dir_right = int(self.snake_dir == (1, 0))
        dir_down = int(self.snake_dir == (0, 1))
        dir_left = int(self.snake_dir == (-1, 0))
        
        # Danger detection in all 8 directions (more comprehensive)
        dangers = []
        directions = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        
        for dx, dy in directions:
            next_x, next_y = head_x + dx, head_y + dy
            is_danger = (next_x < 0 or next_x >= self.width or 
                        next_y < 0 or next_y >= self.height or 
                        (next_x, next_y) in self.snake_pos)
            dangers.append(int(is_danger))
        
        # Food direction (relative to head)
        food_dir_x = np.sign(food_x - head_x) if food_x != head_x else 0
        food_dir_y = np.sign(food_y - head_y) if food_y != head_y else 0
        
        # Normalized distances
        food_dist_x = abs(food_x - head_x) / self.width
        food_dist_y = abs(food_y - head_y) / self.height
        manhattan_dist = (abs(food_x - head_x) + abs(food_y - head_y)) / (self.width + self.height)
        
        # Snake properties
        snake_length_norm = len(self.snake_pos) / (self.width * self.height)
        
        # Tail direction (helps avoid trapping itself)
        if len(self.snake_pos) > 1:
            tail_x, tail_y = self.snake_pos[-1]
            tail_dir_x = np.sign(tail_x - head_x) if tail_x != head_x else 0
            tail_dir_y = np.sign(tail_y - head_y) if tail_y != head_y else 0
        else:
            tail_dir_x = tail_dir_y = 0
        
        # Available space in each direction (look ahead)
        space_counts = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # up, right, down, left
            count = 0
            x, y = head_x, head_y
            for _ in range(max(self.width, self.height)):
                x += dx
                y += dy
                if (x < 0 or x >= self.width or y < 0 or y >= self.height or 
                    (x, y) in self.snake_pos):
                    break
                count += 1
            space_counts.append(count / max(self.width, self.height))
        
        # Combine all features
        state = [
            # Direction (4 features)
            dir_up, dir_right, dir_down, dir_left,
            
            # Dangers in 8 directions (8 features)
            *dangers,
            
            # Food information (5 features)
            food_dir_x, food_dir_y, food_dist_x, food_dist_y, manhattan_dist,
            
            # Snake properties (3 features)
            snake_length_norm, tail_dir_x, tail_dir_y,
            
            # Available space (4 features)
            *space_counts
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute action with improved reward system"""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_r:
                    self.reset()
        
        if self.paused:
            return self._get_state(), 0, False
        
        # Store previous position for reward calculation
        prev_head = self.snake_pos[0]
        prev_food_dist = abs(prev_head[0] - self.food_pos[0]) + abs(prev_head[1] - self.food_pos[1])
        
        # Update direction based on action
        self._update_direction(action)
        
        # Move snake
        head_x, head_y = self.snake_pos[0]
        new_head = (head_x + self.snake_dir[0], head_y + self.snake_dir[1])
        
        self.steps_without_food += 1
        self.total_steps += 1
        
        # Check collisions
        new_x, new_y = new_head
        collision = (new_x < 0 or new_x >= self.width or 
                    new_y < 0 or new_y >= self.height or
                    new_head in self.snake_pos)
        
        # Timeout check
        timeout = self.steps_without_food >= self.max_steps_without_food
        
        if collision or timeout:
            self.game_over = True
            reward = -100 if collision else -50  
            return self._get_state(), reward, True
        
        # Move snake
        self.snake_pos.insert(0, new_head)
        
        # Calculate reward
        reward = self._calculate_reward(prev_food_dist)
        
        # Check if food eaten
        if new_head == self.food_pos:
            self.score += 1
            self.steps_without_food = 0
            # Create particle effect
            self._create_food_eaten_particles(new_head)
            # Create score popup
            self._create_score_popup(new_head, 100 + (len(self.snake_pos) ** 1.5))
            # Exponential reward based on snake length
            reward += 100 + (len(self.snake_pos) ** 1.5)
            self.place_food()
        else:
            # Remove tail if no food eaten
            self.snake_pos.pop()
        
        # Update particles
        self._update_particles()
        
        # Update score popups
        self._update_score_popups()
        
        # Update display
        self._update_display()
        
        return self._get_state(), reward, self.game_over
    
    def _update_direction(self, action):
        """Update direction based on action (0: straight, 1: right, 2: left)"""
        if action == 1:  # Turn right
            direction_map = {
                (0, -1): (1, 0),   # Up -> Right
                (1, 0): (0, 1),    # Right -> Down
                (0, 1): (-1, 0),   # Down -> Left
                (-1, 0): (0, -1)   # Left -> Up
            }
            self.snake_dir = direction_map[self.snake_dir]
        elif action == 2:  # Turn left
            direction_map = {
                (0, -1): (-1, 0),  # Up -> Left
                (1, 0): (0, -1),   # Right -> Up
                (0, 1): (1, 0),    # Down -> Right
                (-1, 0): (0, 1)    # Left -> Down
            }
            self.snake_dir = direction_map[self.snake_dir]
        # action == 0: continue straight (no change)
    
    def _calculate_reward(self, prev_food_dist):
        head_x, head_y = self.snake_pos[0]
        food_x, food_y = self.food_pos
        current_food_dist = abs(head_x - food_x) + abs(head_y - food_y)
        
        # Base survival reward
        reward = 1
        
        # Movement toward/away from food
        if current_food_dist < prev_food_dist:
            reward += 5  # Reward for getting closer
        elif current_food_dist > prev_food_dist:
            reward -= 3  # Penalty for moving away
        
        # Penalty for staying in corners/edges (encourages exploration)
        if head_x == 0 or head_x == self.width-1 or head_y == 0 or head_y == self.height-1:
            reward -= 2
        
        # Small penalty for long episodes without progress
        if self.steps_without_food > self.max_steps_without_food * 0.7:
            reward -= 5
        
        return reward

    def _update_display(self):
        """Enhanced visual display with modern UI elements"""
        # Update animation variables
        self.food_pulse = (self.food_pulse + 0.2) % (2 * np.pi)
        self.snake_glow = (self.snake_glow + 0.15) % (2 * np.pi)
        
        # Create gradient background
        self._draw_gradient_background()
        
        # Draw enhanced grid
        self._draw_enhanced_grid()
        
        # Draw snake with advanced effects
        self._draw_enhanced_snake()
        
        # Draw enhanced food
        self._draw_enhanced_food()
        
        # Draw particles
        self._draw_particles()
        
        # Draw score popups
        self._draw_score_popups()
        
        # Draw UI elements
        self._draw_ui_elements()
        
        # Draw help overlay if needed
        if self.show_help:
            self._draw_help_overlay()
        
        # Draw pause overlay if needed
        if self.paused:
            self._draw_pause_overlay()
        
        # Draw game over overlay if needed
        if self.game_over:
            self._draw_game_over_overlay()
        
        pygame.display.update()
        self.clock.tick(self.speed)
    
    def _draw_gradient_background(self):
        """Draw a subtle gradient background"""
        for y in range(self.display_height):
            # Create a subtle gradient from top to bottom
            intensity = int(15 + (y / self.display_height) * 10)
            color = (intensity, intensity, intensity + 10)
            pygame.draw.line(self.display, color, (0, y), (self.display_width, y))
    
    def _draw_enhanced_grid(self):
        """Draw an enhanced grid with subtle lines"""
        for x in range(0, self.display_width, self.block_size):
            pygame.draw.line(self.display, self.light_gray, (x, 0), (x, self.display_height), 1)
        for y in range(0, self.display_height, self.block_size):
            pygame.draw.line(self.display, self.light_gray, (0, y), (self.display_width, y), 1)
    
    def _draw_enhanced_snake(self):
        """Draw snake with advanced visual effects"""
        # Draw snake trail first (behind the snake)
        if len(self.snake_pos) > 1:
            for i, pos in enumerate(self.snake_pos[1:], 1):
                x, y = pos[0] * self.block_size, pos[1] * self.block_size
                trail_alpha = max(20, 100 - (i * 8))
                trail_surface = pygame.Surface((self.block_size, self.block_size), pygame.SRCALPHA)
                trail_color = (0, 100, 0, trail_alpha)
                pygame.draw.rect(trail_surface, trail_color, [0, 0, self.block_size, self.block_size])
                self.display.blit(trail_surface, (x, y))
        
        for i, pos in enumerate(self.snake_pos):
            x, y = pos[0] * self.block_size, pos[1] * self.block_size
            
            if i == 0:  # Head
                # Main head
                head_color = self.green
                pygame.draw.rect(self.display, head_color, [
                    x + 1, y + 1, self.block_size - 2, self.block_size - 2
                ])
                
                # Glowing effect around head
                glow_intensity = int(50 + 30 * np.sin(self.snake_glow))
                glow_color = (0, glow_intensity, 0)
                pygame.draw.rect(self.display, glow_color, [
                    x, y, self.block_size, self.block_size
                ], 2)
                
                # Eyes
                eye_size = 3
                eye_offset = 4
                pygame.draw.circle(self.display, self.black, 
                                 (x + eye_offset, y + eye_offset), eye_size)
                pygame.draw.circle(self.display, self.black, 
                                 (x + self.block_size - eye_offset, y + eye_offset), eye_size)
                
                # Tongue effect
                tongue_length = 6
                tongue_x = x + self.block_size // 2
                tongue_y = y + self.block_size
                if self.snake_dir == (0, 1):  # Moving down
                    pygame.draw.line(self.display, self.red, 
                                   (tongue_x, tongue_y), 
                                   (tongue_x, tongue_y + tongue_length), 2)
                elif self.snake_dir == (0, -1):  # Moving up
                    pygame.draw.line(self.display, self.red, 
                                   (tongue_x, y), 
                                   (tongue_x, y - tongue_length), 2)
                elif self.snake_dir == (1, 0):  # Moving right
                    pygame.draw.line(self.display, self.red, 
                                   (x + self.block_size, tongue_y), 
                                   (x + self.block_size + tongue_length, tongue_y), 2)
                elif self.snake_dir == (-1, 0):  # Moving left
                    pygame.draw.line(self.display, self.red, 
                                   (x, tongue_y), 
                                   (x - tongue_length, tongue_y), 2)
                
            else:  # Body
                # Gradient body color based on position
                intensity = max(50, 255 - (i * 20))
                body_color = (0, intensity, 0)
                
                # Main body segment
                pygame.draw.rect(self.display, body_color, [
                    x + 2, y + 2, self.block_size - 4, self.block_size - 4
                ])
                
                # Subtle border
                border_color = (0, max(30, intensity - 30), 0)
                pygame.draw.rect(self.display, border_color, [
                    x + 2, y + 2, self.block_size - 4, self.block_size - 4
                ], 1)
                
                # Add some texture to body segments
                if i % 2 == 0:
                    highlight_color = (0, min(255, intensity + 30), 0)
                    pygame.draw.circle(self.display, highlight_color, 
                                     (x + self.block_size // 3, y + self.block_size // 3), 2)
    
    def _draw_enhanced_food(self):
        """Draw food with pulsing and glowing effects"""
        x, y = self.food_pos[0] * self.block_size, self.food_pos[1] * self.block_size
        
        # Pulsing effect
        pulse_size = int(2 + np.sin(self.food_pulse) * 2)
        
        # Multiple glow layers for depth
        for i in range(3):
            glow_alpha = 100 - (i * 30)
            glow_size = self.block_size // 2 + pulse_size + (i * 3)
            glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            glow_color = (255, 100, 100, glow_alpha)
            pygame.draw.circle(glow_surface, glow_color, (glow_size, glow_size), glow_size)
            self.display.blit(glow_surface, (x + self.block_size // 2 - glow_size, y + self.block_size // 2 - glow_size))
        
        # Main food with gradient
        food_center_x = x + self.block_size // 2
        food_center_y = y + self.block_size // 2
        food_radius = self.block_size // 2 - 2
        
        # Create gradient effect
        for r in range(food_radius, 0, -1):
            intensity = int(255 - (food_radius - r) * 3)
            color = (intensity, max(0, intensity - 100), max(0, intensity - 100))
            pygame.draw.circle(self.display, color, (food_center_x, food_center_y), r)
        
        # Highlight
        highlight_color = (255, 200, 200)
        pygame.draw.circle(self.display, highlight_color, 
                         (food_center_x - 2, food_center_y - 2), 
                         food_radius // 3)
        
        # Sparkle effect
        sparkle_angle = self.food_pulse * 2
        sparkle_x = food_center_x + int(np.cos(sparkle_angle) * food_radius * 0.7)
        sparkle_y = food_center_y + int(np.sin(sparkle_angle) * food_radius * 0.7)
        pygame.draw.circle(self.display, self.yellow, (sparkle_x, sparkle_y), 2)

    def _create_score_popup(self, position, points):
        """Create a score popup effect"""
        x, y = position[0] * self.block_size + self.block_size // 2, position[1] * self.block_size
        popup = {
            'x': x,
            'y': y,
            'text': f'+{int(points)}',
            'life': 60,
            'max_life': 60,
            'vy': -1,  # Move upward
            'alpha': 255
        }
        self.score_popups.append(popup)
    
    def _update_score_popups(self):
        """Update score popup positions and life"""
        for popup in self.score_popups[:]:
            popup['y'] += popup['vy']
            popup['life'] -= 1
            popup['alpha'] = int(255 * (popup['life'] / popup['max_life']))
            
            if popup['life'] <= 0:
                self.score_popups.remove(popup)
    
    def _draw_score_popups(self):
        """Draw all active score popups"""
        for popup in self.score_popups:
            if popup['alpha'] > 0:
                # Create text surface with alpha
                text_surface = pygame.Surface((100, 30), pygame.SRCALPHA)
                text_color = (255, 255, 100, popup['alpha'])
                text = self.score_font.render(popup['text'], True, text_color)
                text_rect = text.get_rect(center=(50, 15))
                text_surface.blit(text, text_rect)
                
                # Draw with shadow effect
                shadow_surface = pygame.Surface((100, 30), pygame.SRCALPHA)
                shadow_color = (0, 0, 0, popup['alpha'] // 2)
                shadow = self.score_font.render(popup['text'], True, shadow_color)
                shadow_rect = shadow.get_rect(center=(52, 17))
                shadow_surface.blit(shadow, shadow_rect)
                
                self.display.blit(shadow_surface, (popup['x'] - 50, popup['y'] - 15))
                self.display.blit(text_surface, (popup['x'] - 50, popup['y'] - 15))
    
    def _draw_ui_elements(self):
        """Draw enhanced UI elements with modern design"""
        # Background panel for stats
        panel_width = 200
        panel_height = 120
        panel_x = 10
        panel_y = 10
        
        # Semi-transparent panel background
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(180)
        panel_surface.fill(self.dark_gray)
        self.display.blit(panel_surface, (panel_x, panel_y))
        
        # Panel border
        pygame.draw.rect(self.display, self.light_gray, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Score display
        score_text = self.score_font.render(f'SCORE: {self.score}', True, self.yellow)
        score_rect = score_text.get_rect(topleft=(panel_x + 15, panel_y + 15))
        self.display.blit(score_text, score_rect)
        
        # Length display
        length_text = self.info_font.render(f'Length: {len(self.snake_pos)}', True, self.white)
        length_rect = length_text.get_rect(topleft=(panel_x + 15, panel_y + 45))
        self.display.blit(length_text, length_rect)
        
        # Steps display
        steps_text = self.info_font.render(f'Steps: {self.steps_without_food}', True, self.white)
        steps_rect = steps_text.get_rect(topleft=(panel_x + 15, panel_y + 65))
        self.display.blit(steps_text, steps_rect)
        
        # Controls hint
        controls_text = self.small_font.render('SPACE: Pause | H: Help | R: Reset', True, self.light_gray)
        controls_rect = controls_text.get_rect(bottomleft=(10, self.display_height - 10))
        self.display.blit(controls_text, controls_rect)
    
    def _draw_help_overlay(self):
        """Draw help overlay with game instructions"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.display_width, self.display_height))
        overlay.set_alpha(200)
        overlay.fill(self.black)
        self.display.blit(overlay, (0, 0))
        
        # Help panel
        panel_width = 400
        panel_height = 300
        panel_x = (self.display_width - panel_width) // 2
        panel_y = (self.display_height - panel_height) // 2
        
        # Panel background
        pygame.draw.rect(self.display, self.dark_gray, 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.display, self.light_gray, 
                        (panel_x, panel_y, panel_width, panel_height), 3)
        
        # Title
        title_text = self.title_font.render('GAME CONTROLS', True, self.yellow)
        title_rect = title_text.get_rect(center=(self.display_width // 2, panel_y + 40))
        self.display.blit(title_text, title_rect)
        
        # Help text
        help_lines = [
            'SPACE - Pause/Resume Game',
            'H - Toggle Help',
            'R - Reset Game',
            '',
            'OBJECTIVE:',
            'Eat food to grow longer',
            'Avoid walls and yourself',
            'Get the highest score!',
            '',
            'Press H to close'
        ]
        
        for i, line in enumerate(help_lines):
            if line:
                color = self.yellow if line == 'OBJECTIVE:' else self.white
                text = self.info_font.render(line, True, color)
                text_rect = text.get_rect(center=(self.display_width // 2, panel_y + 80 + i * 25))
                self.display.blit(text, text_rect)
    
    def _draw_pause_overlay(self):
        """Draw pause overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.display_width, self.display_height))
        overlay.set_alpha(150)
        overlay.fill(self.black)
        self.display.blit(overlay, (0, 0))
        
        # Pause text
        pause_text = self.title_font.render('PAUSED', True, self.yellow)
        pause_rect = pause_text.get_rect(center=(self.display_width // 2, self.display_height // 2))
        self.display.blit(pause_text, pause_rect)
        
        # Resume hint
        resume_text = self.info_font.render('Press SPACE to resume', True, self.white)
        resume_rect = resume_text.get_rect(center=(self.display_width // 2, self.display_height // 2 + 40))
        self.display.blit(resume_text, resume_rect)

    def _create_food_eaten_particles(self, position):
        """Create particle effect when food is eaten"""
        x, y = position[0] * self.block_size + self.block_size // 2, position[1] * self.block_size + self.block_size // 2
        
        for _ in range(8):
            particle = {
                'x': x,
                'y': y,
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(-3, 3),
                'life': 30,
                'max_life': 30,
                'color': (random.randint(200, 255), random.randint(100, 200), random.randint(50, 150))
            }
            self.food_eaten_particles.append(particle)
    
    def _update_particles(self):
        """Update and remove expired particles"""
        # Update food eaten particles
        for particle in self.food_eaten_particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                self.food_eaten_particles.remove(particle)
    
    def _draw_particles(self):
        """Draw all active particles"""
        for particle in self.food_eaten_particles:
            alpha = int(255 * (particle['life'] / particle['max_life']))
            size = int(3 * (particle['life'] / particle['max_life']))
            
            if size > 0:
                # Create a surface for the particle with alpha
                particle_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                particle_color = (*particle['color'], alpha)
                pygame.draw.circle(particle_surface, particle_color, (size, size), size)
                self.display.blit(particle_surface, (particle['x'] - size, particle['y'] - size))

    def _draw_game_over_overlay(self):
        """Draw game over overlay with final score and options"""
        overlay = pygame.Surface((self.display_width, self.display_height))
        overlay.set_alpha(200)
        overlay.fill(self.black)
        self.display.blit(overlay, (0, 0))

        panel_width = 400
        panel_height = 250
        panel_x = (self.display_width - panel_width) // 2
        panel_y = (self.display_height - panel_height) // 2

        pygame.draw.rect(self.display, self.dark_gray, (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.display, self.light_gray, (panel_x, panel_y, panel_width, panel_height), 3)

        title_text = self.title_font.render('GAME OVER', True, self.red)
        title_rect = title_text.get_rect(center=(self.display_width // 2, panel_y + 40))
        self.display.blit(title_text, title_rect)

        score_text = self.score_font.render(f'Final Score: {self.score}', True, self.yellow)
        score_rect = score_text.get_rect(center=(self.display_width // 2, panel_y + 80))
        self.display.blit(score_text, score_rect)

        controls_text = self.small_font.render('Press SPACE to play again or R to reset', True, self.white)
        controls_rect = controls_text.get_rect(center=(self.display_width // 2, panel_y + 120))
        self.display.blit(controls_text, controls_rect)


# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


# Improved DQN Agent with Double DQN and Prioritized Experience Replay
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, hidden_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.tau = 0.005  # Soft update parameter
        
        # Neural networks (Double DQN)
        self.q_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.update_frequency = 4
        self.target_update_frequency = 100
        self.step_count = 0
        
        # Tracking metrics
        self.scores = []
        self.avg_scores = []
        self.losses = []
        self.epsilons = []
        self.episode_lengths = []
        self.q_values = []
    
    def get_action(self, state):
        """Epsilon-greedy action selection with neural network"""
        self.step_count += 1
        
        if np.random.random() > self.epsilon:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                self.q_values.append(q_values.max().item())
                return q_values.argmax().item()
        else:
            return random.randint(0, self.action_size - 1)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Soft update target network
        if self.step_count % self.target_update_frequency == 0:
            self.soft_update()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def soft_update(self):
        """Soft update of target network"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, filepath):
        """Save model and training data"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': self.scores,
            'losses': self.losses
        }, filepath)
    
    def load_model(self, filepath):
        """Load model and training data"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.scores = checkpoint.get('scores', [])
            self.losses = checkpoint.get('losses', [])
            return True
        return False


def train_agent(episodes=2000, speed=25, checkpoint_freq=200, save_dir="models"):
    """Enhanced training with better monitoring"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    env = SnakeGameEnv(speed=speed)
    state_size = len(env._get_state())  # Dynamic state size
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    print(f"State size: {state_size}")
    print(f"Training on device: {agent.device}")
    
    best_avg_score = -float('inf')
    training_start_time = datetime.now()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.game_over:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if agent.step_count % agent.update_frequency == 0:
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Track metrics
        agent.scores.append(env.score)
        agent.episode_lengths.append(steps)
        agent.epsilons.append(agent.epsilon)
        
        # Calculate rolling average
        window_size = min(100, len(agent.scores))
        avg_score = np.mean(agent.scores[-window_size:])
        agent.avg_scores.append(avg_score)
        
        # Print progress
        if episode % 20 == 0:
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
            avg_q = np.mean(agent.q_values[-100:]) if agent.q_values else 0
            print(f"Episode {episode:4d} | Score: {env.score:2d} | Avg: {avg_score:5.2f} | "
                  f"Steps: {steps:3d} | ε: {agent.epsilon:.3f} | Loss: {avg_loss:.4f} | "
                  f"Q-val: {avg_q:.2f}")
        
        # Save best model
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save_model(os.path.join(save_dir, "best_model.pth"))
        
        # Checkpoint saves
        if episode % checkpoint_freq == 0 and episode > 0:
            agent.save_model(os.path.join(save_dir, f"checkpoint_{episode}.pth"))
            plot_training_progress(agent, episode)
            
            # Save training log
            training_log = {
                'episode': episode,
                'scores': agent.scores,
                'avg_scores': agent.avg_scores,
                'losses': agent.losses,
                'epsilons': agent.epsilons,
                'episode_lengths': agent.episode_lengths,
                'best_avg_score': best_avg_score,
                'training_time': str(datetime.now() - training_start_time)
            }
            
            with open(os.path.join(save_dir, f"training_log_{episode}.json"), 'w') as f:
                json.dump(training_log, f, indent=2)
    
    # Final save
    agent.save_model(os.path.join(save_dir, "final_model.pth"))
    plot_training_progress(agent, episodes)
    
    return agent


def plot_training_progress(agent, episode):
    """Enhanced plotting with more metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Scores
    axes[0, 0].plot(agent.scores, alpha=0.6, color='blue', label='Raw Score')
    if len(agent.avg_scores) > 0:
        axes[0, 0].plot(agent.avg_scores, color='red', linewidth=2, label='Avg Score (100 ep)')
    axes[0, 0].set_title('Training Scores')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(agent.episode_lengths, alpha=0.7, color='green')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Epsilon decay
    axes[0, 2].plot(agent.epsilons, color='orange')
    axes[0, 2].set_title('Exploration Rate (ε)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Epsilon')
    axes[0, 2].grid(True)
    
    # Training loss
    if agent.losses:
        # Smooth the loss curve
        window_size = min(100, len(agent.losses))
        if len(agent.losses) >= window_size:
            smoothed_loss = np.convolve(agent.losses, np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(smoothed_loss, color='purple')
        else:
            axes[1, 0].plot(agent.losses, color='purple')
    axes[1, 0].set_title('Training Loss (Smoothed)')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].grid(True)
    
    # Q-values
    if agent.q_values:
        window_size = min(100, len(agent.q_values))
        if len(agent.q_values) >= window_size:
            smoothed_q = np.convolve(agent.q_values, np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(smoothed_q, color='brown')
        else:
            axes[1, 1].plot(agent.q_values, color='brown')
    axes[1, 1].set_title('Average Q-Values')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Q-Value')
    axes[1, 1].grid(True)
    
    # Score distribution
    if len(agent.scores) > 50:
        axes[1, 2].hist(agent.scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 2].axvline(np.mean(agent.scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(agent.scores):.2f}')
        axes[1, 2].set_title('Score Distribution')
        axes[1, 2].set_xlabel('Score')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/training_progress_ep{episode}.png', dpi=150, bbox_inches='tight')
    plt.close()


def test_agent(episodes=10, speed=3, model_path="models/best_model.pth", render=True):
    """Test the trained agent"""
    env = SnakeGameEnv(speed=speed)
    state_size = len(env._get_state())
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    if agent.load_model(model_path):
        agent.epsilon = 0.01  # Minimal exploration
        print(f"Loaded model from {model_path}")
    else:
        print(f"Could not load model from {model_path}")
        return
    
    scores = []
    steps_list = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.game_over:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render:
                time.sleep(0.1)  # Slow down for visualization
            
            if done:
                break
        
        scores.append(env.score)
        steps_list.append(steps)
        print(f"Test Episode {episode + 1:2d}: Score = {env.score:2d}, Steps = {steps:3d}, Reward = {total_reward:6.1f}")
    
    print(f"\n--- Test Results ---")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Best Score: {max(scores)}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    
    
    return scores, steps_list


if __name__ == "__main__":
    print("=== Enhanced Snake RL Training ===")
    print("Training improved agent with Deep Q-Network...")
    
    # Train the agent
    agent = train_agent(episodes=100, speed=25, checkpoint_freq=50)
    
    print("\n=== Testing Trained Agent ===")
    # Test the trained agent
    test_agent(episodes=40, speed=10, render=True)