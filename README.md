# Quantum Simulator

A multi-week project to better understand quantum principles.

## Project Files

- **main.py**: Main file to run the simulator.
- **qsim.py**: Core quantum simulation logic.
- **gui.py**: GUI control for the simulator.

## Dependencies

```bash
pip install numpy matplotlib PyQt5 qutip
```

## Running the Simulator

```bash
python main.py
```

## Known Bugs

- The simulator somtimes crashes when running multiple sequential simulations.

## Features

- Qubit simulator for up to 5 qubits.
- Classical and quantum superpositions states: Including decoherence switch, magnetic field interactions, and entanglement entropies.
- Visualization and export results with value plots, Bloch spheres, and data.
- Randomize, reset, and manually adjust simulation settings.
- Efficient Qthreading simulation using PyQt5 and QuTiP.