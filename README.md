
# 🐍  Snake Game - Deep Reinforcement Learning

A sophisticated Snake game implementation with advanced Deep Q-Learning (DQN) and modern UI enhancements, built with Pygame and PyTorch.

## ✨ Features

### 🎮 **Enhanced Gameplay**
- **Modern UI Design**: Dark theme with gradient backgrounds and smooth animations
- **Particle Effects**: Explosion effects when food is eaten
- **Score Popups**: Animated score displays with shadow effects
- **Snake Trail**: Semi-transparent trail behind the snake
- **Visual Enhancements**: Glowing effects, eyes, tongue, and body textures

### 🤖 **Advanced AI**
- **Double DQN**: Improved learning stability with target networks
- **Experience Replay**: Efficient memory management for better learning
- **Prioritized Learning**: Smart batch sampling for optimal training
- **Soft Updates**: Smooth target network updates for stable learning

### 🎨 **Visual Effects**
- **Dynamic Animations**: Smooth pulsing, glowing, and sparkle effects
- **Modern Typography**: Multiple font sizes with professional styling
- **Interactive Overlays**: Help system, pause screen, and game over display
- **Color Gradients**: Smooth transitions and depth effects

### 📊 **Training & Monitoring**
- **Real-time Metrics**: Live training progress with multiple charts
- **Checkpoint System**: Automatic model saving and recovery
- **Performance Analytics**: Comprehensive training statistics
- **Visualization Tools**: Matplotlib integration for progress tracking

## 🚀 Quick Start

### Prerequisites
```bash
pip install pygame torch numpy matplotlib
```

### Run the Game
```bash
python snake_rl.py
```

### Controls
- **SPACE**: Pause/Resume game
- **H**: Toggle help overlay
- **R**: Reset game
- **Close Window**: Quit game

## 🏗️ Architecture

### **Core Components**
1. **SnakeGameEnv**: Enhanced game environment with modern UI
2. **DQN**: Deep Q-Network with dropout and ReLU activation
3. **DQNAgent**: Advanced agent with experience replay and target networks
4. **Training System**: Comprehensive training with checkpointing

### **Neural Network Structure**
```
Input (24 features) → Hidden (256) → Hidden (256) → Hidden (128) → Output (3 actions)
```

### **State Representation**
- **Direction Encoding**: 4 features (up, right, down, left)
- **Danger Detection**: 8 features (8-direction collision detection)
- **Food Information**: 5 features (direction, distance, position)
- **Snake Properties**: 3 features (length, tail direction)
- **Available Space**: 4 features (space in each direction)

## 📈 Training Process

### **Hyperparameters**
- **Learning Rate**: 0.001
- **Discount Factor (γ)**: 0.99
- **Epsilon Decay**: 0.995
- **Batch Size**: 64
- **Memory Size**: 50,000 experiences

### **Reward System**
- **Survival**: +1 per step
- **Food Collection**: +100 + (snake_length^1.5)
- **Movement Toward Food**: +5
- **Movement Away from Food**: -3
- **Collision**: -100
- **Timeout**: -50

## 🎯 Performance

### **Training Results**
- **Episodes**: Configurable (default: 100)
- **Checkpoint Frequency**: Every 50 episodes
- **Model Saving**: Automatic best model preservation
- **Progress Tracking**: Real-time score and loss monitoring

### **Visual Outputs**
- **Training Plots**: Score progression, episode lengths, epsilon decay
- **Loss Curves**: Smoothed training loss visualization
- **Q-Value Analysis**: Average Q-value tracking
- **Score Distribution**: Histogram of achieved scores

## 🔧 Customization

### **Game Settings**
```python
# Adjust game speed and size
env = SnakeGameEnv(width=20, height=20, block_size=20, speed=5)

# Modify training parameters
agent = train_agent(episodes=100, speed=25, checkpoint_freq=50)
```

### **AI Parameters**
```python
# Customize learning parameters
agent = DQNAgent(state_size=24, action_size=3, lr=0.001, hidden_size=256)
```

## 📁 Project Structure
```
rl/
├── snake_rl.py          # Main game and training code
├── models/              # Saved model checkpoints
│   ├── best_model.pth
│   ├── checkpoint_*.pth
│   └── final_model.pth
├── plots/               # Training progress visualizations
│   └── training_progress_ep*.png
└── README.md            # This file
```

## 🎮 Game Modes

### **Training Mode**
- **Speed**: 25 FPS (configurable)
- **AI Learning**: Active reinforcement learning
- **Real-time Display**: Live training visualization
- **Checkpointing**: Automatic progress saving

### **Testing Mode**
- **Speed**: 10 FPS (configurable)
- **AI Play**: Trained model demonstration
- **Performance Metrics**: Score and step analysis
- **Visual Effects**: Full UI enhancement showcase

## 🔬 Technical Details

### **Reinforcement Learning**
- **Algorithm**: Deep Q-Learning with Experience Replay
- **Network**: 4-layer fully connected neural network
- **Optimizer**: Adam with gradient clipping
- **Loss Function**: Mean Squared Error (MSE)

### **Performance Optimizations**
- **GPU Support**: Automatic CUDA detection and utilization
- **Memory Management**: Efficient experience replay buffer
- **Batch Processing**: Optimized neural network updates
- **Visual Rendering**: Hardware-accelerated graphics

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating feature branches
3. Implementing improvements
4. Submitting pull requests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Pygame**: Game development framework
- **PyTorch**: Deep learning framework
- **Matplotlib**: Data visualization
- **OpenAI Gym**: RL environment inspiration

---

**Enjoy playing and training your AI Snake! 🐍✨**

*Built with ❤️ and lots of 🧠*
