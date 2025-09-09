# 🐍 Snake Puzzle Solver Dashboard

A Streamlit web application for creating, generating, and solving Snake (also called Tunnel) logic puzzles using Mixed Integer Programming (MIP) via the [snake-mip-solver](https://github.com/DenHvideDvaerg/snake-mip-solver) library.

## 🎯 Features

- **Interactive Puzzle Creation**: Click-and-create interface for designing custom puzzles
- **Random Puzzle Generation**: Generate puzzles with configurable difficulty and seed values
- **Constraint Management**: Set row/column constraints with visual feedback
- **MIP-based Solving**: Uses advanced mathematical optimization for guaranteed solutions
- **Performance Metrics**: Real-time solving statistics and timing information

## 🚀 Getting Started

### Online Demo

Visit the live demo on Streamlit Community Cloud: **[https://snake-dashboard.streamlit.app/](https://snake-dashboard.streamlit.app/)** 

### Running Locally

1. Clone this repository:
```bash
git clone https://github.com/DenHvideDvaerg/snake-dashboard
cd snake-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## 🧩 About Snake Puzzles

Snake puzzles are path-finding logic puzzles where you must:
- Connect start and end points with a continuous path
- Fill the exact number of cells specified by row/column constraints  
- Ensure the snake never touches itself (including diagonally)
- Create a single, unbroken path from head to tail

## 🔧 Technical Details

- **Framework**: Streamlit for interactive web interface
- **Solver Engine**: [snake-mip-solver](https://github.com/DenHvideDvaerg/snake-mip-solver) using Mixed Integer Programming (MIP)
- **Optimization Backend**: OR-Tools for high-performance mathematical solving
- **Solution Method**: Finds feasible solutions that satisfy all puzzle constraints
- **UI Styling**: CSS with Streamlit theme adaptation for dark/light mode support

## 🔗 Links

- **Solver Library**: https://github.com/DenHvideDvaerg/snake-mip-solver
- **Play Online**: https://gridpuzzle.com/snake
- **Streamlit**: https://streamlit.io

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
