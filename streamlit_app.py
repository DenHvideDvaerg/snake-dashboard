""" 
Streamlit app for creating and solving Snake puzzles.
"""

import streamlit as st
from snake_mip_solver import SnakePuzzle, SnakeSolver, SnakePuzzleGenerator
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Snake Puzzle Solver",
    page_icon="🐍",
    layout="wide"
)

def initialize_session_state():
    if "num_rows" not in st.session_state:
        st.session_state.num_rows = 8
    if "num_cols" not in st.session_state:
        st.session_state.num_cols = 8
    if "row_sums" not in st.session_state:
        st.session_state.row_sums = [None] * st.session_state.num_rows
    if "col_sums" not in st.session_state:
        st.session_state.col_sums = [None] * st.session_state.num_cols
    if "puzzle_grid" not in st.session_state:
        # Initialize with a random puzzle
        generate_random_puzzle()

def render_puzzle_editor():
    """Render the interactive grid for puzzle creation."""
    
    col_dimensions, col_generator = st.columns([1, 2])

    # Grid size controls
    with col_dimensions:
        st.subheader("✏️ Puzzle Editor")
        col_rows, col_columns = st.columns([1, 1])
        with col_rows:
            new_rows = st.number_input(
                "Rows",
                min_value=3,
                max_value=20,
                value=st.session_state.num_rows,
                key="new_num_rows"
            )

        with col_columns:
            new_cols = st.number_input(
                "Cols", 
                min_value=3,
                max_value=20,
                value=st.session_state.num_cols,
                key="new_num_cols"
            )
        
        # Handle grid size changes
        if new_rows != st.session_state.num_rows or new_cols != st.session_state.num_cols:
            st.session_state.num_rows = new_rows
            st.session_state.num_cols = new_cols
            # Resize arrays
            st.session_state.row_sums = [None] * new_rows
            st.session_state.col_sums = [None] * new_cols
            st.session_state.puzzle_grid = [[False for _ in range(new_cols)] for _ in range(new_rows)]
            st.rerun()

    with col_generator:
        render_random_generation_options()

    st.markdown("### 📊 Constraint Grid")
    st.markdown("Set the number of filled cells required for each row and column. Check cells to mark start and end snake positions.")

    # Create constraint grid with visual feedback
    num_rows = st.session_state.num_rows
    num_cols = st.session_state.num_cols
    
    # Column headers with constraint inputs
    header_cols = st.columns([1] + [1] * num_cols, gap="small")
    
    for col in range(num_cols):
        with header_cols[col + 1]:
            # Create options for selectbox (None + numbers 0 to num_rows)
            options = ["None"] + list(range(0, num_rows + 1))
            current_value = st.session_state.col_sums[col]
            
            # Find the index of current value in options
            if current_value is None:
                index = 0  # "None" is at index 0
            else:
                index = options.index(current_value)
            
            selected = st.selectbox(
                f"C{col}",
                options=options,
                index=index,
                key=f"col_sum_{col}",
                help=f"Required filled cells in column {col} (None = unconstrained)"
            )
            
            # Convert back to None or int
            st.session_state.col_sums[col] = None if selected == "None" else selected

    # Grid rows with row constraints and cells
    for row in range(num_rows):
        row_cols = st.columns([1] + [1] * num_cols, gap="small")
        
        # Row constraint input
        with row_cols[0]:
            # Create options for selectbox (None + numbers 0 to num_cols)
            options = ["None"] + list(range(0, num_cols + 1))
            current_value = st.session_state.row_sums[row]
            
            # Find the index of current value in options
            if current_value is None:
                index = 0  # "None" is at index 0
            else:
                index = options.index(current_value)
            
            selected = st.selectbox(
                f"R{row}",
                options=options,
                index=index,
                key=f"row_sum_{row}",
                help=f"Required filled cells in row {row} (None = unconstrained)"
            )
            
            # Convert back to None or int
            st.session_state.row_sums[row] = None if selected == "None" else selected
        
        # Grid cells
        for col in range(num_cols):
            with row_cols[col + 1]:
                cell_key = f"grid_cell_{row}_{col}"
                label = f"dummy_{row}_{col}"
                # Regular cell - checkbox for start/end
                st.session_state.puzzle_grid[row][col] = st.checkbox(
                    label,
                    value=st.session_state.puzzle_grid[row][col],
                    key=cell_key,
                    help=f"Cell ({row}, {col})",
                    label_visibility="collapsed"
                )
    
    # Calculate start and end positions from checked cells
    checked_cells = [(r, c) for r in range(num_rows) for c in range(num_cols) 
                     if st.session_state.puzzle_grid[r][c]]

    # Validate exactly 2 selected cells for proper start/end positioning
    if len(checked_cells) == 2:
        # Sort cells by row first, then by column for consistent assignment
        # Start = first cell (top-left most), End = second cell (bottom-right most)
        sorted_cells = sorted(checked_cells)  # Sorts by (row, col) tuple naturally
        start_cell = sorted_cells[0]  # Earlier position (smaller row/col)
        end_cell = sorted_cells[1]    # Later position (larger row/col)
        
        st.session_state.start_cell = start_cell
        st.session_state.end_cell = end_cell
        st.success(f"✅ Start: {start_cell}, End: {end_cell}")
    else:
        st.error(f"❌ {len(checked_cells)} cells selected. Please select exactly 2 cells for start and end positions.")
        st.session_state.start_cell = None
        st.session_state.end_cell = None

    if st.button("🗑️ Clear Puzzle", type="secondary"):
        st.session_state.puzzle_grid = [[False for _ in range(st.session_state.num_cols)] 
                                        for _ in range(st.session_state.num_rows)]
        st.session_state.row_sums = [None] * st.session_state.num_rows
        st.session_state.col_sums = [None] * st.session_state.num_cols
        st.rerun()
    
    
def create_and_solve_puzzle():
    """Create SnakePuzzle object and solve it."""
    try:
        # Use constraints as-is (None means unconstrained, 0+ are valid constraint values)
        row_sums = st.session_state.row_sums
        col_sums = st.session_state.col_sums
        
        puzzle = SnakePuzzle(
            row_sums=row_sums,
            col_sums=col_sums, 
            start_cell=st.session_state.start_cell,
            end_cell=st.session_state.end_cell
        )
        
        # Store puzzle in session state for other components
        st.session_state.current_puzzle = puzzle
        st.session_state.solving = True
        
        with st.spinner("🔍 Solving puzzle..."):
            solver = SnakeSolver(puzzle)
            
            # Start timing
            start_time = time.time()
            solution = solver.solve(verbose=True)
            end_time = time.time()
            
            # Calculate solve time
            solve_time = end_time - start_time
            st.session_state.solve_time = solve_time
            
        if solution:
            st.session_state.current_solution = solution
            st.toast(f"✅ Solution found with {len(solution)} filled cells in {solve_time:.2f}s!")
        else:
            st.toast("❌ No solution found for this puzzle configuration.")
            st.session_state.current_solution = None
            st.session_state.solve_time = None
            
    except Exception as e:
        st.error(f"Error creating/solving puzzle: {str(e)}")
    finally:
        st.session_state.solving = False


def render_random_generation_options():
    """Render random puzzle generation options in a modal-like interface."""
    st.markdown("### 🎲 Random Puzzle Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fill_percentage = st.slider(
            "Fill Percentage",
            min_value=0.05,
            max_value=0.55,
            value=0.4,
            step=0.05,
            help="Percentage of cells to fill in the generated puzzle (0.05 = 5%, 0.55 = 55%)"
        )
    
    with col2:
        use_seed = st.checkbox("Use Random Seed", value=False, help="Set a specific seed for reproducible puzzles")
        
        if use_seed:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=999999,
                value=42,
                help="Seed value for random generation (same seed = same puzzle)"
            )
        else:
            random_seed = None
    
    if st.button("✨ Generate Puzzle", type="primary", use_container_width=True):
        generate_random_puzzle(fill_percentage=fill_percentage, random_seed=random_seed)


def generate_random_puzzle(fill_percentage=0.4, random_seed=None):
    """Generate a random puzzle using SnakePuzzleGenerator."""
    try:
        with st.spinner("🎲 Generating random puzzle..."):
            generator = SnakePuzzleGenerator(random_seed)
            
            # Start timing
            start_time = time.time()
            puzzle, generated_solution = generator.generate(
                rows=st.session_state.num_rows,
                cols=st.session_state.num_cols,
                fill_percentage=fill_percentage
            )
            end_time = time.time()
            
            # Calculate generation time
            generation_time = end_time - start_time
            
        # Update session state with generated puzzle
        st.session_state.row_sums = puzzle.row_sums
        st.session_state.col_sums = puzzle.col_sums
        
        # Clear the puzzle grid first
        st.session_state.puzzle_grid = [[False for _ in range(st.session_state.num_cols)] 
                                       for _ in range(st.session_state.num_rows)]
        
        
        start_cell = puzzle.get_start_cell()
        end_cell = puzzle.get_end_cell()
        st.session_state.start_cell = start_cell
        st.session_state.end_cell = end_cell
        st.session_state.puzzle_grid[start_cell[0]][start_cell[1]] = True
        st.session_state.puzzle_grid[end_cell[0]][end_cell[1]] = True

        # Store the puzzle and solution
        st.session_state.current_puzzle = puzzle
        st.session_state.current_solution = None  # Clear solution until solved
        
        # Success message with generation info
        seed_info = f" (seed: {random_seed})" if random_seed is not None else ""
        
        # Format generation time
        if generation_time < 1:
            time_display = f"{generation_time*1000:.0f}ms"
        else:
            time_display = f"{generation_time:.2f}s"
            
        st.toast(f"✅ Random puzzle generated in {time_display}! Fill: {fill_percentage:.0%}{seed_info}")
        
    except Exception as e:
        st.error(f"Error generating puzzle: {str(e)}")


def create_snake_board_html(puzzle=None, solution=None, show_constraints=True):
    """Create HTML representation of Snake puzzle board."""
    if puzzle is None and "current_puzzle" not in st.session_state:
        return None
    
    if puzzle is None:
        puzzle = st.session_state.current_puzzle
    
    num_rows = puzzle.rows if puzzle else st.session_state.num_rows
    num_cols = puzzle.cols if puzzle else st.session_state.num_cols
    
    # CSS styling that adapts to Streamlit's theme
    html = """
    <style>
    .snake-grid {
        display: inline-block;
        border: 2px solid currentColor;
        font-family: monospace;
        font-size: 14px;
        margin: 10px auto;
        border-collapse: collapse;
    }
    .snake-grid td {
        width: 40px;
        height: 40px;
        text-align: center;
        vertical-align: middle;
        border: 1px solid currentColor;
        position: relative;
        opacity: 0.8;
    }
    .snake-grid .constraint-cell {
        background: rgba(var(--primary-color-rgb, 255, 75, 75), 0.1);
        font-weight: bold;
        font-size: 12px;
        width: 30px;
        height: 30px;
    }
    .snake-grid .empty { 
        background: rgba(128, 128, 128, 0.1);
        opacity: 0.6;
    }
    .snake-grid .filled { 
        background: rgba(var(--primary-color-rgb, 255, 75, 75), 0.3);
        font-weight: bold;
        opacity: 1;
    }
    .snake-grid .endpoint { 
        background: rgba(var(--primary-color-rgb, 255, 75, 75), 0.4);
        color: currentColor;
        font-weight: bold;
        font-size: 16px;
        opacity: 1;
    }

    </style>
    """
    
    # Start building the table
    html += '<table class="snake-grid">'
    
    if show_constraints:
        # Header row with column constraints
        html += "<tr>"
        html += '<td class="constraint-cell"></td>'  # Empty corner
        
        for col in range(num_cols):
            col_constraint = puzzle.col_sums[col] if puzzle and puzzle.col_sums else st.session_state.col_sums[col]
            
            # Display constraint (show "−" for None)
            display_value = "−" if col_constraint is None else str(col_constraint)
            html += f'<td class="constraint-cell">{display_value}</td>'
        
        html += "</tr>"
    
    # Grid rows
    for row in range(num_rows):
        html += "<tr>"
        
        if show_constraints:
            # Row constraint
            row_constraint = puzzle.row_sums[row] if puzzle and puzzle.row_sums else st.session_state.row_sums[row]
            
            # Display constraint (show "−" for None)
            display_value = "−" if row_constraint is None else str(row_constraint)
            html += f'<td class="constraint-cell">{display_value}</td>'
        
        # Grid cells
        for col in range(num_cols):
            classes = []
            cell_content = ""
            
            # Get start and end cells from puzzle or session state
            start_cell = None
            end_cell = None
            if puzzle:
                start_cell = puzzle.start_cell
                end_cell = puzzle.end_cell
            elif "start_cell" in st.session_state and "end_cell" in st.session_state:
                start_cell = st.session_state.start_cell
                end_cell = st.session_state.end_cell
            
            # Check cell state and assign content + classes
            if start_cell and (row, col) == start_cell:
                cell_content = "■"  # Square for endpoint
                classes.append("endpoint")
            elif end_cell and (row, col) == end_cell:
                cell_content = "■"  # Same square for endpoint
                classes.append("endpoint")
            elif solution and (row, col) in solution:
                # Regular filled cell (not start/end)
                cell_content = "●"  # Circle for body segments
                classes.append("filled")
            else:
                cell_content = "·"
                classes.append("empty")
            
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            html += f"<td{class_attr}>{cell_content}</td>"
        
        html += "</tr>"
    
    html += "</table>"
    return html


def display_snake_board(puzzle=None, solution=None, title="Snake Puzzle", show_constraints=True):
    """Display the Snake puzzle board using HTML styling."""
    st.subheader(title)
    
    # Create styled display of the Snake board
    board_html = create_snake_board_html(puzzle, solution, show_constraints)
    if board_html is None:
        st.warning("Unable to display the Snake board. No puzzle is available.")
        return
    
    st.markdown(board_html, unsafe_allow_html=True)


def render_solution_grid():
    """Render the solution visualization if available."""
    if "current_solution" not in st.session_state or st.session_state.current_solution is None:
        st.info("🔍 No solution to display. Create and solve a puzzle first.")
        return
    
    solution = st.session_state.current_solution
    puzzle = st.session_state.get("current_puzzle")
    
    if puzzle:
        # Display beautiful HTML board
        display_snake_board(puzzle, solution, "🎯 Solution", show_constraints=True)
            
        # Display solution as text format (like in the examples)
        with st.expander("📋 View Text Representation"):
            try:
                board_text = puzzle.get_board_visualization(solution, show_indices=False)
                st.code(board_text, language=None)
            except Exception as e:
                st.error(f"Error generating board visualization: {str(e)}")
                

def render_sidebar():
    """Render the sidebar with puzzle introduction, instructions and links."""
    with st.sidebar:
        st.header("📚 About")
        st.markdown("""
        **Snake** (also called Tunnel) is a spatial awareness path-finding logic puzzle. 
        The goal is to connect the snake's head with its tail by filling in grid cells 
        following these rules:

        🔗 **Single connected path** - The snake forms one continuous line from head to tail
        
        🚫 **No self-touching** - The snake's body never touches itself, not even diagonally
        
        🔢 **Row/Column constraints** - Numbers outside the grid tell you how many cells 
        must be filled in each row and column
        
        🧭 **Movement rules** - The snake's body can only be filled horizontally and vertically
        
        
        This app uses the **[snake-mip-solver](https://github.com/DenHvideDvaerg/snake-mip-solver)** package, which uses Mixed Integer Programming (MIP) to solve the puzzle.
        """)
        
        st.markdown("---")
        
        st.header("🎮 How to Use")
        st.markdown("""
        1. **Generate a random puzzle** with variable fill percentage
        2. **Or manually adjust** grid size and row/column constraints
        3. **Select exactly 2 cells** to mark start and end positions
        4. **Click 'Solve Puzzle'** to find the solution
        """) 
        
        st.markdown("---")
        
        st.header("🔗 Useful Links")
        st.markdown("""
        - 🔧 **[Solver](https://github.com/DenHvideDvaerg/snake-mip-solver)** - The solver powering this app
        - 🌐 **[Snake Puzzles Online](https://gridpuzzle.com/snake)** - Play more puzzles in your browser
        """)


def render_visualization():
    """Render puzzle visualization (with and without solution)."""
    # Show current puzzle state if available
    if "current_puzzle" in st.session_state:
        st.subheader("Visualization")
        puzzle = st.session_state.current_puzzle
        
        # Show puzzle board (with or without solution)
        if "current_solution" in st.session_state and st.session_state.current_solution:
            # Show solution
            render_solution_grid()
        else:
            # Show puzzle setup without solution
            display_snake_board(puzzle, None, "🧩 Current Puzzle Setup", show_constraints=True)
        
        # Solver info if solving
        if st.session_state.get("solving", False):
            st.info("🔄 Solving in progress...")

def render_summary():
    """Render puzzle summary and information."""
    if "current_puzzle" in st.session_state:
        st.subheader("📋 Summary")
        puzzle = st.session_state.current_puzzle
        
        # Puzzle info
        with st.expander("ℹ️ Puzzle Information", expanded=True):
            st.write(f"**Grid Size:** {puzzle.rows}×{puzzle.cols}")
            st.write(f"**Start Position:** {puzzle.start_cell}")
            st.write(f"**End Position:** {puzzle.end_cell}")
            st.write(f"**Row Constraints:** {puzzle.row_sums}")
            st.write(f"**Column Constraints:** {puzzle.col_sums}")
            
            # Calculate total expected filled cells
            total_expected = sum(s for s in puzzle.row_sums if s is not None)
            st.write(f"**Expected filled cells:** {total_expected}")
    else:
        st.info("No puzzle loaded yet.")


def render_solution_statistics():
    """Render solution statistics if available."""
    if "current_puzzle" in st.session_state and "current_solution" in st.session_state and st.session_state.current_solution:
        puzzle = st.session_state.current_puzzle
        solution = st.session_state.current_solution
        
        st.markdown("**📊 Solution Statistics:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filled Cells", len(solution))
        with col2:
            is_valid = puzzle.is_valid_solution(solution)
            st.metric("Valid Solution", "✅ Yes" if is_valid else "❌ No")
        with col3:
            solve_time = st.session_state.get("solve_time")
            if solve_time is not None:
                if solve_time < 1:
                    time_display = f"{solve_time*1000:.0f}ms"
                else:
                    time_display = f"{solve_time:.2f}s"
                st.metric("Solve Time", time_display)
            else:
                st.metric("Solve Time", "N/A")
  
  


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    st.title("🐍 Snake Puzzle Solver")
    st.markdown("Create, solve, and visualize Snake logic puzzles using mathematical optimization.")

    # Render sidebar
    render_sidebar()
    
    col_input, col_visualization = st.columns([2, 1])

    with col_input:
        render_puzzle_editor()
    
    with col_visualization:
        render_summary()
        if st.button("🔧 Solve Puzzle", type="primary"):
            create_and_solve_puzzle()
        render_visualization()
        render_solution_statistics()



if __name__ == "__main__":
    main()