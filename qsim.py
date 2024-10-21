import numpy as np
from qutip import basis, tensor, mesolve, sigmax, sigmay, sigmaz, qeye, expect, entropy_vn


class QSim:
    def __init__(self, num_qubits, gamma, B_fields, initial_states, T1, T2):
        self.num_qubits = num_qubits
        self.gamma = gamma
        self.B_fields = B_fields
        self.initial_states = initial_states
        self.T1 = T1
        self.T2 = T2

        self.construct_hamiltonian()
        self.construct_initial_state()
        self.construct_collapse_operators()

        self.time_points = []
        self.expectations = [[] for _ in range(num_qubits)]
        self.entropies = []

    def construct_hamiltonian(self):
        H = 0
        for i in range(self.num_qubits):
            Bx, By, Bz = self.B_fields[i]
            gamma_i = self.gamma[i]
            H_i = gamma_i * (Bx * sigmax() + By * sigmay() + Bz * sigmaz())
            ops = [qeye(2) for _ in range(self.num_qubits)]
            ops[i] = H_i
            H += tensor(ops)
        self.H = H

    def construct_initial_state(self):
        psi_list = []
        for psi in self.initial_states:
            psi_qobj = psi[0] * basis(2, 0) + psi[1] * basis(2, 1)
            psi_list.append(psi_qobj)
        self.psi0_qobj = tensor(psi_list)

    def construct_collapse_operators(self):
        c_ops = []
        for i in range(self.num_qubits):
            ops = [qeye(2) for _ in range(self.num_qubits)]
            if self.T1[i] > 0:
                rate = 1 / self.T1[i]
                ops[i] = np.sqrt(rate) * sigmax()
                c_ops.append(tensor(ops))
            if self.T2[i] > 0:
                rate = 1 / (2 * self.T2[i])
                ops[i] = np.sqrt(rate) * sigmaz()
                c_ops.append(tensor(ops))
        self.c_ops = c_ops

    def evolve(self, dt):
        t_list = [0, dt]
        result = mesolve(self.H, self.psi0_qobj, t_list, c_ops=self.c_ops, options={'store_states': True, 'atol': 1e-8, 'rtol': 1e-6})
        self.psi0_qobj = result.states[-1]

        # Record data
        current_time = self.time_points[-1] + dt if self.time_points else dt
        self.time_points.append(current_time)
        for i in range(self.num_qubits):
            expectations_i = {}
            ops = [qeye(2) for _ in range(self.num_qubits)]

            ops[i] = sigmax() # σx
            expectations_i['x'] = expect(tensor(ops), self.psi0_qobj)
            ops[i] = sigmay() # σy
            expectations_i['y'] = expect(tensor(ops), self.psi0_qobj)
            ops[i] = sigmaz() # σz
            expectations_i['z'] = expect(tensor(ops), self.psi0_qobj)

            self.expectations[i].append(expectations_i)

        entropies = []
        for i in range(self.num_qubits):
            rho_red = self.psi0_qobj.ptrace(i)
            entropy = entropy_vn(rho_red)
            entropies.append(entropy)
        self.entropies.append(entropies)

    def save_data(self, file_path):
        with open(file_path, 'w') as f:
            headers = ['Time']
            num_qubits = self.num_qubits
            for i in range(num_qubits):
                headers += [f'Qubit_{i + 1}_x', f'Qubit_{i + 1}_y', f'Qubit_{i + 1}_z']
            for i in range(num_qubits):
                headers += [f'Qubit_{i + 1}_Entropy']
            f.write(','.join(headers) + '\n')

            num_points = len(self.time_points)
            for idx in range(num_points):
                row = [f"{self.time_points[idx]:.5f}"]
                for i in range(num_qubits):
                    exp = self.expectations[i][idx]
                    row += [f"{exp['x']:.5f}", f"{exp['y']:.5f}", f"{exp['z']:.5f}"]
                for i in range(num_qubits):
                    row += [f"{self.entropies[idx][i]:.5f}"]
                f.write(','.join(row) + '\n')
