import random

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox, QTabWidget, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qutip import Bloch

from qsim import QSim


class QSimGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Quantum Simulator')
        self.setGeometry(100, 100, 1200, 800)
        self.simulation_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left panel
        control_layout = QVBoxLayout()
        self.init_controls(control_layout)
        main_layout.addLayout(control_layout, 1)

        # Right panel
        self.plot_tabs = QTabWidget()
        self.init_plots()
        main_layout.addWidget(self.plot_tabs, 2)

    def init_controls(self, layout):
        font = QFont()
        font.setPointSize(10)

        # Qubits Count
        num_qubits_layout = QHBoxLayout()
        num_qubits_label = QLabel("Number of Qubits:")
        num_qubits_label.setFont(font)
        self.num_qubits_spin = QSpinBox()
        self.num_qubits_spin.setRange(1, 5)
        self.num_qubits_spin.setValue(1)
        self.num_qubits_spin.valueChanged.connect(self.reset_simulation_environment)
        num_qubits_layout.addWidget(num_qubits_label)
        num_qubits_layout.addWidget(self.num_qubits_spin)
        layout.addLayout(num_qubits_layout)

        # Unique Qubit Controls
        self.qubit_controls_layout = QVBoxLayout()
        layout.addLayout(self.qubit_controls_layout)
        self.qubit_controls = []
        self.update_qubit_controls()

        # Simulation Parameters
        sim_params_label = QLabel("Simulation Parameters:")
        sim_params_label.setFont(font)
        layout.addWidget(sim_params_label)

        # Time Step
        dt_layout = QHBoxLayout()
        dt_label = QLabel("Time Step (dt):")
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(1e-5, 1.0)
        self.dt_spin.setDecimals(5)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setValue(0.01)
        dt_layout.addWidget(dt_label)
        dt_layout.addWidget(self.dt_spin)
        layout.addLayout(dt_layout)

        # Step Count
        steps_layout = QHBoxLayout()
        steps_label = QLabel("Number of Steps:")
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 10000)
        self.steps_spin.setSingleStep(100)
        self.steps_spin.setValue(1000)
        steps_layout.addWidget(steps_label)
        steps_layout.addWidget(self.steps_spin)
        layout.addLayout(steps_layout)

        # Run and Stop Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        # Reset and Randomize Buttons
        action_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset Simulation")
        self.reset_button.clicked.connect(self.reset_simulation_environment)
        self.randomize_button = QPushButton("Randomize Settings")
        self.randomize_button.clicked.connect(self.randomize_settings)
        action_layout.addWidget(self.reset_button)
        action_layout.addWidget(self.randomize_button)
        layout.addLayout(action_layout)

        # Export Data Buttons
        export_layout = QHBoxLayout()
        self.export_data_button = QPushButton("Export Data")
        self.export_data_button.clicked.connect(self.export_data)
        self.export_plots_button = QPushButton("Export Plots")
        self.export_plots_button.clicked.connect(self.export_plots)
        export_layout.addWidget(self.export_data_button)
        export_layout.addWidget(self.export_plots_button)
        layout.addLayout(export_layout)

        layout.addStretch()

    def update_qubit_controls(self):
        # Reset
        for i in reversed(range(self.qubit_controls_layout.count())):
            widget_to_remove = self.qubit_controls_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
        self.qubit_controls = []

        num_qubits = self.num_qubits_spin.value()
        font = QFont()
        font.setPointSize(9)

        for i in range(num_qubits):
            qubit_group = QWidget()
            qubit_layout = QGridLayout()
            qubit_group.setLayout(qubit_layout)

            qubit_label = QLabel(f"Qubit {i + 1} Controls:")
            qubit_label.setFont(font)
            qubit_layout.addWidget(qubit_label, 0, 0, 1, 2)

            # Initial State
            init_state_label = QLabel("Initial State:")
            init_state_combo = QComboBox()
            init_state_combo.addItems(['Classical One', 'Classical Zero', 'Quantum Plus', 'Quantum Minus'])
            qubit_layout.addWidget(init_state_label, 1, 0)
            qubit_layout.addWidget(init_state_combo, 1, 1)

            # Gamma
            gamma_label = QLabel("Gamma:")
            gamma_spin = QDoubleSpinBox()
            gamma_spin.setRange(0.1, 5.0)
            gamma_spin.setSingleStep(0.1)
            gamma_spin.setValue(1.0)
            qubit_layout.addWidget(gamma_label, 2, 0)
            qubit_layout.addWidget(gamma_spin, 2, 1)

            # Magnetic Field
            Bx_label = QLabel("Bx:")
            Bx_spin = QDoubleSpinBox()
            Bx_spin.setRange(-5.0, 5.0)
            Bx_spin.setSingleStep(0.1)
            Bx_spin.setValue(0.0)
            qubit_layout.addWidget(Bx_label, 3, 0)
            qubit_layout.addWidget(Bx_spin, 3, 1)

            By_label = QLabel("By:")
            By_spin = QDoubleSpinBox()
            By_spin.setRange(-5.0, 5.0)
            By_spin.setSingleStep(0.1)
            By_spin.setValue(0.0)
            qubit_layout.addWidget(By_label, 4, 0)
            qubit_layout.addWidget(By_spin, 4, 1)

            Bz_label = QLabel("Bz:")
            Bz_spin = QDoubleSpinBox()
            Bz_spin.setRange(-5.0, 5.0)
            Bz_spin.setSingleStep(0.1)
            Bz_spin.setValue(1.0)
            qubit_layout.addWidget(Bz_label, 5, 0)
            qubit_layout.addWidget(Bz_spin, 5, 1)

            # Decoherence
            decoherence_check = QCheckBox("Include Decoherence")
            qubit_layout.addWidget(decoherence_check, 6, 0, 1, 2)

            T1_label = QLabel("T1:")
            T1_spin = QDoubleSpinBox()
            T1_spin.setRange(0.1, 10.0)
            T1_spin.setSingleStep(0.1)
            T1_spin.setValue(1.0)
            T1_spin.setEnabled(False)
            qubit_layout.addWidget(T1_label, 7, 0)
            qubit_layout.addWidget(T1_spin, 7, 1)

            T2_label = QLabel("T2:")
            T2_spin = QDoubleSpinBox()
            T2_spin.setRange(0.1, 10.0)
            T2_spin.setSingleStep(0.1)
            T2_spin.setValue(1.0)
            T2_spin.setEnabled(False)
            qubit_layout.addWidget(T2_label, 8, 0)
            qubit_layout.addWidget(T2_spin, 8, 1)

            def toggle_decoherence(state, T1_spin=T1_spin, T2_spin=T2_spin):
                T1_spin.setEnabled(state == Qt.Checked)
                T2_spin.setEnabled(state == Qt.Checked)

            decoherence_check.stateChanged.connect(toggle_decoherence)

            self.qubit_controls_layout.addWidget(qubit_group)
            self.qubit_controls.append({'init_state': init_state_combo, 'gamma': gamma_spin, 'Bx': Bx_spin, 'By': By_spin, 'Bz': Bz_spin, 'decoherence': decoherence_check, 'T1': T1_spin, 'T2': T2_spin})

    def init_plots(self):
        # Clear existing tabs
        while self.plot_tabs.count():
            widget = self.plot_tabs.widget(0)
            self.plot_tabs.removeTab(0)
            widget.deleteLater()

        # Expectation Values Plot
        self.expect_fig = Figure()
        self.expect_canvas = FigureCanvas(self.expect_fig)
        self.expect_ax = self.expect_fig.add_subplot(111)
        self.plot_tabs.addTab(self.expect_canvas, "Expectation Values")

        # Entanglement Entropy Plot
        self.entropy_fig = Figure()
        self.entropy_canvas = FigureCanvas(self.entropy_fig)
        self.entropy_ax = self.entropy_fig.add_subplot(111)
        self.plot_tabs.addTab(self.entropy_canvas, "Entanglement Entropy")

        # Bloch Spheres
        self.bloch_tabs = QTabWidget()
        self.blochs = []
        num_qubits = self.num_qubits_spin.value()

        for i in range(num_qubits):
            bloch_fig = Figure()
            bloch_canvas = FigureCanvas(bloch_fig)
            bloch_ax = bloch_fig.add_subplot(111, projection='3d')
            bloch = Bloch(axes=bloch_ax)
            self.blochs.append(bloch)
            self.bloch_tabs.addTab(bloch_canvas, f"Bloch Sphere Qubit {i + 1}")

        if num_qubits > 0:
            self.plot_tabs.addTab(self.bloch_tabs, "Bloch Spheres")

    def run_simulation(self):
        if self.simulation_thread and self.simulation_thread.isRunning():
            QMessageBox.warning(self, "Simulation Running", "A simulation is already running.")
            return
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.randomize_button.setEnabled(False)
        self.simulation_thread = SimulationThread(self)
        self.simulation_thread.update_plots.connect(self.update_plots)
        self.simulation_thread.finished.connect(self.simulation_finished)
        self.simulation_thread.error_occurred.connect(self.handle_simulation_error)
        self.simulation_thread.start()

    def stop_simulation(self):
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.wait()
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.reset_button.setEnabled(True)
            self.randomize_button.setEnabled(True)

    def reset_simulation_environment(self):
        if self.simulation_thread and self.simulation_thread.isRunning():
            QMessageBox.warning(self, "Simulation Running", "Cannot reset while simulation is running.")
            return
        self.update_qubit_controls()
        self.init_plots()
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.randomize_button.setEnabled(True)

    def randomize_settings(self):
        if self.simulation_thread and self.simulation_thread.isRunning():
            QMessageBox.warning(self, "Simulation Running", "Cannot randomize settings while simulation is running.")
            return

        # Randomize number of qubits
        num_qubits = random.randint(1, 5)
        self.num_qubits_spin.setValue(num_qubits)
        self.update_qubit_controls()

        # Randomize per-qubit settings
        for qubit in self.qubit_controls:
            # Random initial state
            init_state = random.choice(['Classical One', 'Classical Zero', 'Quantum Plus', 'Quantum Minus'])
            index = qubit['init_state'].findText(init_state, Qt.MatchFixedString)

            if index >= 0:
                qubit['init_state'].setCurrentIndex(index)

            # Random gamma
            gamma = round(random.uniform(0.1, 5.0), 2)
            qubit['gamma'].setValue(gamma)

            # Random magnetic field components
            Bx = round(random.uniform(-5.0, 5.0), 2)
            By = round(random.uniform(-5.0, 5.0), 2)
            Bz = round(random.uniform(-5.0, 5.0), 2)
            qubit['Bx'].setValue(Bx)
            qubit['By'].setValue(By)
            qubit['Bz'].setValue(Bz)

            # Random decoherence
            decoherence = random.choice([True, False])
            qubit['decoherence'].setChecked(decoherence)
            qubit['T1'].setEnabled(decoherence)
            qubit['T2'].setEnabled(decoherence)
            if decoherence:
                T1 = round(random.uniform(0.1, 10.0), 2)
                T2 = round(random.uniform(0.1, 10.0), 2)
                qubit['T1'].setValue(T1)
                qubit['T2'].setValue(T2)

        # Randomize simulation parameters
        dt = round(random.uniform(0.001, 0.1), 5)
        num_steps = random.randint(100, 2000)
        self.dt_spin.setValue(dt)
        self.steps_spin.setValue(num_steps)

        # Reset plots
        self.init_plots()

        QMessageBox.information(self, "Randomization Complete", "Simulation settings have been randomized.")

    def update_plots(self, data):
        time_points = data['time']
        expectations = data['expectation']
        entropies = data['entropy']

        num_qubits = len(expectations)

        # Update Expectation Values Plot
        self.expect_ax.clear()
        colors = plt.cm.get_cmap('tab10', num_qubits)
        for i in range(num_qubits):
            color = colors(i)
            x_values = [exp['x'] for exp in expectations[i]]
            y_values = [exp['y'] for exp in expectations[i]]
            z_values = [exp['z'] for exp in expectations[i]]
            self.expect_ax.plot(time_points, x_values, label=f'Qubit {i + 1} ⟨σx⟩', color=color, linestyle='-')
            self.expect_ax.plot(time_points, y_values, label=f'Qubit {i + 1} ⟨σy⟩', color=color, linestyle='--')
            self.expect_ax.plot(time_points, z_values, label=f'Qubit {i + 1} ⟨σz⟩', color=color, linestyle=':')
        self.expect_ax.set_xlabel('Time')
        self.expect_ax.set_ylabel('Expectation Values')
        self.expect_ax.set_title('Expectation Values Over Time')
        self.expect_ax.legend(loc='upper right', fontsize='small', ncol=2)
        self.expect_ax.grid(True)
        self.expect_canvas.draw()

        # Update Entanglement Entropy Plot
        self.entropy_ax.clear()
        for i in range(num_qubits):
            entropies_i = [entropy[i] for entropy in entropies]
            self.entropy_ax.plot(time_points, entropies_i, label=f'Qubit {i + 1}')
        self.entropy_ax.set_xlabel('Time')
        self.entropy_ax.set_ylabel('Entropy')
        self.entropy_ax.set_title('Entanglement Entropy Over Time')
        self.entropy_ax.legend()
        self.entropy_ax.grid(True)
        self.entropy_canvas.draw()

        # Update Bloch Spheres
        for i in range(num_qubits):
            bloch = self.blochs[i]
            bloch.clear()
            bloch.add_vectors([expectations[i][-1]['x'], expectations[i][-1]['y'], expectations[i][-1]['z']])
            bloch.render()
            bloch.fig.canvas.draw()

    def simulation_finished(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.randomize_button.setEnabled(True)
        QMessageBox.information(self, "Simulation Finished", "The simulation has completed.")

    def handle_simulation_error(self, error_message):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.randomize_button.setEnabled(True)
        QMessageBox.critical(self, "Simulation Error", f"An error occurred during simulation:\n{error_message}")

    def export_data(self):
        if not hasattr(self, 'simulation_thread') or not self.simulation_thread:
            QMessageBox.warning(self, "No Data", "No simulation data to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            self.simulation_thread.simulation.save_data(file_path)
            QMessageBox.information(self, "Export Successful", "Data exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def export_plots(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        try:
            self.expect_fig.savefig(f"{directory}/expectation_values.png")
            self.entropy_fig.savefig(f"{directory}/entanglement_entropy.png")
            for i, bloch in enumerate(self.blochs):
                bloch.fig.savefig(f"{directory}/bloch_sphere_qubit_{i + 1}.png")
            QMessageBox.information(self, "Export Successful", "Plots exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))


class SimulationThread(QThread):
    update_plots = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self._running = True

    def run(self):
        # Retrieve simulation parameters
        num_qubits = self.gui.num_qubits_spin.value()
        gamma = [self.gui.qubit_controls[i]['gamma'].value() for i in range(num_qubits)]
        B_fields = [(self.gui.qubit_controls[i]['Bx'].value(), self.gui.qubit_controls[i]['By'].value(),  self.gui.qubit_controls[i]['Bz'].value()) for i in range(num_qubits)]
        dt = self.gui.dt_spin.value()
        num_steps = self.gui.steps_spin.value()

        T1 = []
        T2 = []
        initial_states = []
        for i in range(num_qubits):
            decoh = self.gui.qubit_controls[i]['decoherence'].isChecked()
            T1_i = self.gui.qubit_controls[i]['T1'].value() if decoh else 0.0
            T2_i = self.gui.qubit_controls[i]['T2'].value() if decoh else 0.0
            T1.append(T1_i)
            T2.append(T2_i)

            # Initial state
            init_state = self.gui.qubit_controls[i]['init_state'].currentText()
            if init_state == 'Classical Zero':
                psi0_i = np.array([1, 0])
            elif init_state == 'Classical One':
                psi0_i = np.array([0, 1])
            elif init_state == 'Quantum Plus':
                psi0_i = np.array([1, 1]) / np.sqrt(2)
            elif init_state == 'Quantum Minus':
                psi0_i = np.array([1, -1]) / np.sqrt(2)
            else:
                psi0_i = np.array([1, 0])

            initial_states.append(psi0_i)

        # Initialize simulation
        self.simulation = QSim(num_qubits=num_qubits, gamma=gamma, B_fields=B_fields, initial_states=initial_states, T1=T1, T2=T2)

        for step in range(num_steps):
            if not self._running:
                break
            self.simulation.evolve(dt)
            if step % max(1, num_steps // 100) == 0 or step == num_steps - 1:
                data = {'time': self.simulation.time_points, 'expectation': self.simulation.expectations, 'entropy': self.simulation.entropies}
                self.update_plots.emit(data)
        self._running = False

    def stop(self):
        self._running = False
