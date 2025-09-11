import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class HospitalDoctorAssignmentDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("A Rank-order Assignment Demo (Doctors to Hospitals)")
        self.root.geometry("1024x700")

        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(3, weight=1)

        # title
        title_label = ttk.Label(self.main_frame, text="H-D Assign Demo", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Control Board
        control_frame = ttk.LabelFrame(self.main_frame, text="Control Board", padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        control_frame.columnconfigure(1, weight=1)

        ttk.Button(control_frame, text="Run", command=self.run_demo).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Reset", command=self.reset).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Detailed Results", command=self.show_stats).grid(row=0, column=2, padx=5,
                                                                                         pady=5)

        # All Parameters
        input_frame = ttk.LabelFrame(self.main_frame, text="Settings", padding="5")
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5), pady=5)
        input_frame.columnconfigure(1, weight=1)

        # Doctor count setting
        ttk.Label(input_frame, text="Number of Doctors:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.doctor_count = tk.IntVar(value=10)
        ttk.Spinbox(input_frame, from_=3, to=1000, textvariable=self.doctor_count, width=10).grid(
            row=0, column=1, sticky=tk.W, pady=2, padx=(5, 0))

        # Doctor ranking method setting
        ttk.Label(input_frame, text="Ranking Method:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ranking_method = tk.StringVar(value="random")
        ranking_frame = ttk.Frame(input_frame)
        ranking_frame.grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(ranking_frame, text="Random", variable=self.ranking_method,
                        value="random").pack(side=tk.LEFT)
        ttk.Radiobutton(ranking_frame, text="Custom", variable=self.ranking_method,
                        value="custom").pack(side=tk.LEFT)

        # Hospital count setting
        ttk.Label(input_frame, text="Number of Hospitals:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.hospital_count = tk.IntVar(value=3)
        ttk.Spinbox(input_frame, from_=2, to=500, textvariable=self.hospital_count, width=10).grid(
            row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))

        # Hospital capacity method setting
        ttk.Label(input_frame, text="Capacity Method:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.capacity_method = tk.StringVar(value="random")
        capacity_frame = ttk.Frame(input_frame)
        capacity_frame.grid(row=3, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(capacity_frame, text="Random", variable=self.capacity_method,
                        value="random").pack(side=tk.LEFT)
        ttk.Radiobutton(capacity_frame, text="Custom", variable=self.capacity_method,
                        value="custom").pack(side=tk.LEFT)

        # Custom capacity input frame (hidden by default)
        self.custom_capacity_frame = ttk.Frame(input_frame)
        self.custom_capacity_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.custom_capacity_frame.grid_remove()

        ttk.Label(self.custom_capacity_frame, text="Hospital Capacities:").grid(row=0, column=0, sticky=tk.W)
        self.custom_capacity_vars = []

        # Custom ranking input frame (hidden by default)
        self.custom_ranking_frame = ttk.Frame(input_frame)
        self.custom_ranking_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.custom_ranking_frame.grid_remove()

        ttk.Label(self.custom_ranking_frame, text="Custom Rankings:").grid(row=0, column=0, sticky=tk.W, columnspan=2)
        self.ranking_text = scrolledtext.ScrolledText(self.custom_ranking_frame, width=30, height=5)
        self.ranking_text.grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(self.custom_ranking_frame, text="Load ex.", command=self.load_example_rankings).grid(row=2,
                                                                                                            column=0,
                                                                                                            pady=2)
        ttk.Button(self.custom_ranking_frame, text="Clear", command=self.clear_rankings).grid(row=2, column=1, pady=2)

        # Bind events
        self.capacity_method.trace('w', self.toggle_capacity_input)
        self.ranking_method.trace('w', self.toggle_ranking_input)
        self.hospital_count.trace('w', self.on_hospital_count_change)

        # Result Area
        result_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        result_frame.grid(row=2, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        self.result_text = scrolledtext.ScrolledText(result_frame, width=50, height=15)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Matrix of Preference Area
        detail_frame = ttk.LabelFrame(self.main_frame, text="Preference Matrix", padding="5")
        detail_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5), pady=5)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self.detail_text = scrolledtext.ScrolledText(detail_frame, width=40, height=15)
        self.detail_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # initialize data & demo
        self.doctors = []
        self.hospitals = []
        self.capacities = {}
        self.preferences = {}
        self.assigned = {}
        self.eliminated = []

        # 添加缺失的变量
        self.max_capacity = tk.IntVar(value=4)

        self.generate_data()
        self.update_display()

    # 添加缺失的方法
    def toggle_capacity_input(self, *args):
        """Show or hide custom capacity input"""
        if self.capacity_method.get() == "custom":
            self.custom_capacity_frame.grid()
            self.setup_capacity_inputs()
        else:
            self.custom_capacity_frame.grid_remove()

    def toggle_ranking_input(self, *args):
        """Show or hide custom ranking input"""
        if self.ranking_method.get() == "custom":
            self.custom_ranking_frame.grid()
        else:
            self.custom_ranking_frame.grid_remove()

    def setup_capacity_inputs(self):
        """Setup hospital capacity input fields"""
        # Clear old input fields
        for widget in self.custom_capacity_frame.grid_slaves():
            if int(widget.grid_info()["row"]) >= 1:
                widget.destroy()
        self.custom_capacity_vars = []

        # Create new input fields
        hospital_count = self.hospital_count.get()
        for i in range(hospital_count):
            ttk.Label(self.custom_capacity_frame, text=f"Hospital H{i} Capacity:").grid(
                row=i + 1, column=0, sticky=tk.W, pady=2)
            cap_var = tk.IntVar(value=2)
            self.custom_capacity_vars.append(cap_var)
            ttk.Spinbox(self.custom_capacity_frame, from_=1, to=20, textvariable=cap_var, width=5).grid(
                row=i + 1, column=1, sticky=tk.W, pady=2, padx=(5, 0))

        #Update when count changes
    def on_hospital_count_change(self, *args):
        if self.capacity_method.get() == "custom":
            self.setup_capacity_inputs()
#load example

    def load_example_rankings(self):
        example_text = """D0: H2, H0, H1"""
        self.ranking_text.delete(1.0, tk.END)
        self.ranking_text.insert(1.0, example_text)

    def clear_rankings(self):
        """Clear ranking input"""
        self.ranking_text.delete(1.0, tk.END)

    def generate_data(self):
        # list of D
        doc_count = self.doctor_count.get()
        self.doctors = [f"D{i}" for i in range(doc_count)]

        # list of H
        hosp_count = self.hospital_count.get()
        self.hospitals = [f"H{i}" for i in range(hosp_count)]

        # H capacity
        self.capacities = {}
        if self.capacity_method.get() == "custom" and self.custom_capacity_vars:
            # Use custom capacities
            for i, hosp in enumerate(self.hospitals):
                if i < len(self.custom_capacity_vars):
                    self.capacities[hosp] = self.custom_capacity_vars[i].get()
        else:
            # Randomly assign capacities
            max_cap = 5
            for hosp in self.hospitals:
                cap = np.random.randint(1, max_cap + 1)
                self.capacities[hosp] = cap

        # change until the capacity is enough
        total_capacity = sum(self.capacities.values())
        if total_capacity < doc_count:
            for hosp in self.hospitals:
                if total_capacity >= doc_count:
                    break
                self.capacities[hosp] += 1
                total_capacity += 1

        # H capacity
        self.capacities = {}
        if self.capacity_method.get() == "custom" and self.custom_capacity_vars:
            # Use custom capacities
            for i, hosp in enumerate(self.hospitals):
                if i < len(self.custom_capacity_vars):
                    self.capacities[hosp] = self.custom_capacity_vars[i].get()
        else:
            # Randomly assign capacities
            max_cap = 5
            for hosp in self.hospitals:
                cap = np.random.randint(1, max_cap + 1)
                self.capacities[hosp] = cap


        # Doc Pref
        self.preferences = {}
        if self.ranking_method.get() == "custom":
            self.parse_custom_rankings()
        else:
            for doc in self.doctors:
                # Random assign Doc Pref
                pref = self.hospitals.copy()
                np.random.shuffle(pref)
                self.preferences[doc] = pref

    def parse_custom_rankings(self):
        """Parse custom ranking input"""
        text = self.ranking_text.get(1.0, tk.END).strip()
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if ':' in line:
                parts = line.split(':', 1)
                doc = parts[0].strip()
                if doc in self.doctors:
                    hospitals = [h.strip() for h in parts[1].split(',')]
                    # Only keep existing hospitals
                    valid_hospitals = [h for h in hospitals if h in self.hospitals]
                    self.preferences[doc] = valid_hospitals

        # Add random preferences for missing doctors
        for doc in self.doctors:
            if doc not in self.preferences:
                pref = self.hospitals.copy()
                np.random.shuffle(pref)
                self.preferences[doc] = pref

    def prefs_to_cost(self, doctors, hospitals, preferences, large_penalty=None):
        N, K = len(doctors), len(hospitals)
        if large_penalty is None:
            large_penalty = K
        idx = {h: j for j, h in enumerate(hospitals)}
        cost = np.full((N, K), large_penalty, dtype=float)
        for i, d in enumerate(doctors):
            pref = preferences.get(d, [])
            for r, h in enumerate(pref, start=1):
                j = idx.get(h, None)
                if j is not None:
                    cost[i, j] = r
        return cost

    # better fr (DFS)
    def fr_plus(self, A):
        a, b = np.shape(A)
        r_able_c = [[] for _ in range(a)]
        for i in range(a):
            for j in range(b):
                if A[i, j] == 0:
                    r_able_c[i].append(j)
        c_to_r = [-1] * b

        def DFS(r, dis_c):
            for c in r_able_c[r]:
                if dis_c[c]:
                    continue
                dis_c[c] = True
                if c_to_r[c] == -1 or DFS(c_to_r[c], dis_c):
                    c_to_r[c] = r
                    return True
            return False

        for r in range(a):
            dis_c = [False] * b
            DFS(r, dis_c)

        L = np.full(a, -1, dtype=int)
        for c, r in enumerate(c_to_r):
            if r != -1:
                L[r] = c

        return L

    def hungarian_algorithm(self, cost_matrix, n_rows, n_cols, min_dim):
        max_iter = 42
        iter_count = 0
        A = cost_matrix.copy()

        while iter_count < max_iter:
            result = self.fr_plus(A)

            # Check if we found a complete assignment
            complete_assignment = True
            assigned_count = 0
            for i, col_idx in enumerate(result):
                if col_idx == -1 and i < n_rows:
                    complete_assignment = False
                else:
                    assigned_count += 1

            if complete_assignment and assigned_count == min_dim:
                assignment_matrix = np.zeros((n_rows, n_cols), dtype=int)
                for i, col_idx in enumerate(result):
                    if col_idx != -1 and i < n_rows and col_idx < n_cols:
                        assignment_matrix[i, int(col_idx)] = 1
                return assignment_matrix

            # Find unmatched rows
            unmatched = [i for i in range(n_rows) if i >= len(result) or result[i] == -1]
            if not unmatched:
                break

            # Mark rows and columns for matrix adjustment
            marked_rows = set(unmatched)
            marked_cols = set()

            # Find all reachable rows and columns through alternating paths
            change = True
            while change:
                change = False
                new_rows = []
                for row in marked_rows:
                    for col in range(n_cols):
                        if abs(A[row, col]) < 1e-10 and col not in marked_cols:
                            marked_cols.add(col)
                            change = True
                            # Find rows which use this newly marked column
                            for r in range(n_rows):
                                if r < len(result) and result[r] == col and r not in marked_rows:
                                    new_rows.append(r)
                # Add the new rows to the marked rows
                for row in new_rows:
                    marked_rows.add(row)

            # Find minimum uncovered value
            min_val = float('inf')
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    if row_idx in marked_rows and col_idx not in marked_cols:
                        min_val = min(min_val, A[row_idx, col_idx])

            if min_val == float('inf') or min_val == 0:
                break

            # Adjust matrix: add to covered rows, subtract from uncovered columns
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    if row_idx not in marked_rows:
                        A[row_idx, col_idx] += min_val
                    if col_idx not in marked_cols:
                        A[row_idx, col_idx] -= min_val

            iter_count += 1

        # Return partial assignment if no complete assignment found
        assignment_matrix = np.zeros((n_rows, n_cols), dtype=int)
        for i, col_idx in enumerate(result):
            if col_idx != -1 and i < n_rows and col_idx < n_cols:
                assignment_matrix[i, int(col_idx)] = 1

        return assignment_matrix

    def assignment(self, A, c):
        n_rows, n_cols = np.shape(A)

        # Create expanded matrix by duplicating columns according to capacity
        expanded_A = []
        col_mapping = []

        for col_idx in range(n_cols):
            for _ in range(c[col_idx]):
                expanded_A.append(A[:, col_idx])
                col_mapping.append(col_idx)

        expanded_A = np.array(expanded_A).T

        # Apply Hungarian algorithm with symmetric processing
        exp_n_rows, exp_n_cols = expanded_A.shape
        need_transpose = exp_n_rows > exp_n_cols

        # If n_rows > n_cols, transpose the matrix for processing
        if need_transpose:
            expanded_A = expanded_A.T
            exp_n_rows, exp_n_cols = exp_n_cols, exp_n_rows

        min_dim = exp_n_rows

        # Row reduction: subtract minimum from each row
        for row_idx in range(exp_n_rows):
            min_val = np.min(expanded_A[row_idx])
            expanded_A[row_idx] -= min_val

        # Use Hungarian algorithm
        assignment_matrix = self.hungarian_algorithm(expanded_A, exp_n_rows, exp_n_cols, min_dim)
        # If we transposed the input, transpose the result back
        if need_transpose:
            assignment_matrix = assignment_matrix.T

        # Convert assignment back to original matrix format
        match_results = np.zeros((n_rows, n_cols), dtype=int)
        for row_idx in range(n_rows):
            for col_idx in range(assignment_matrix.shape[1]):
                if assignment_matrix[row_idx, col_idx] == 1 and col_idx < len(col_mapping):
                    original_col = col_mapping[col_idx]
                    match_results[row_idx, original_col] = 1

        return match_results

    def solve_doctor_hospital(self, doctors, hospitals, capacities, preferences, large_penalty=None):
        # Preference
        A = self.prefs_to_cost(doctors, hospitals, preferences, large_penalty=large_penalty)
        # Set of Capacity
        c = np.array([int(capacities.get(h, 0)) for h in hospitals], dtype=int)
        match = self.assignment(A, c)

        # Change the result back to the names
        assigned = {}
        N, K = match.shape
        for i in range(N):
            if match[i].sum() == 0:
                assigned[doctors[i]] = None
            else:
                j = int(np.argmax(match[i]))
                assigned[doctors[i]] = hospitals[j]

        eliminated = [d for d in doctors if assigned.get(d) is None]
        return assigned, eliminated, match

    # Compute the rate about average rank and percentage for each rank
    def compute_metrics(self, doctors, preferences, assigned):
        ranks = []
        for d in doctors:
            h = assigned.get(d)
            if h is None:
                continue
            pref = preferences.get(d, [])
            if h in pref:
                ranks.append(pref.index(h) + 1)

        if not ranks:
            return {"avg_rank": None, "top1": 0.0, "top2": 0.0, "top3": 0.0, "hist": {}}

        ranks = np.array(ranks, dtype=int)
        avg = float(ranks.mean())
        top1 = float((ranks == 1).mean())
        top2 = float((ranks <= 2).mean())
        top3 = float((ranks <= 3).mean())
        uniq, cnt = np.unique(ranks, return_counts=True)
        hist = {int(k): int(v) for k, v in zip(uniq, cnt)}

        return {
            "avg_rank": avg,
            "top1": top1,
            "top2": top2,
            "top3": top3,
            "hist": hist
        }

    # Demo
    def run_demo(self):
        self.status_var.set("Running...")
        self.generate_data()

        try:
            self.assigned, self.eliminated, match_matrix = self.solve_doctor_hospital(
                self.doctors, self.hospitals, self.capacities, self.preferences
            )

            self.update_display()
            self.status_var.set("Complete")

        except Exception as e:
            messagebox.showerror("Error", f": {str(e)}")
            self.status_var.set("Error")

    def reset(self):
        self.generate_data()
        self.assigned = {}
        self.eliminated = []
        self.update_display()
        self.status_var.set("Reset")

    def show_stats(self):
        if not self.assigned:
            messagebox.showwarning("Warning", "Please run the demo first")
            return

        metrics = self.compute_metrics(self.doctors, self.preferences, self.assigned)

        stats_window = tk.Toplevel(self.root)
        stats_window.title("Calculation Results")
        stats_window.geometry("400x300")

        # Create frame
        stats_frame = ttk.Frame(stats_window, padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(stats_frame, text="Calculation Results", font=("Arial", 14, "bold")).pack(pady=10)

        # Show the data
        stats_text = f"Total Doctors: {len(self.doctors)}\n"
        stats_text += f"Successfully Assigned: {len(self.doctors) - len(self.eliminated)}\n"
        stats_text += f"Doctors Not Assigned: {len(self.eliminated)}\n"
        stats_text += f"Average Rank: {metrics['avg_rank']:.2f}\n"
        stats_text += f"First Preference Rate: {metrics['top1'] * 100:.1f}%\n"
        stats_text += f"Top 3 Preference Rate: {metrics['top3'] * 100:.1f}%\n"

        ttk.Label(stats_frame, text=stats_text, justify=tk.LEFT).pack(pady=10)

        ttk.Label(stats_frame, text="Rank Distribution:", font=("Arial", 12, "bold")).pack(pady=(20, 5))

        dist_frame = ttk.Frame(stats_frame)
        dist_frame.pack(fill=tk.X, padx=20)

        for rank, count in metrics['hist'].items():
            ttk.Label(dist_frame, text=f"Rank {rank}: {count} doctors").pack(anchor=tk.W)

    def update_display(self):
        # Clear text areas
        self.result_text.delete(1.0, tk.END)
        self.detail_text.delete(1.0, tk.END)

        # Show hospital capacity information
        self.result_text.insert(tk.END, "Hos Capacity:\n")
        self.result_text.insert(tk.END, "=" * 40 + "\n")
        for hosp in self.hospitals:
            self.result_text.insert(tk.END, f"{hosp}: {self.capacities[hosp]} positions\n")
        self.result_text.insert(tk.END, "\n")

        # Show doctor preferences
        self.result_text.insert(tk.END, "Doc pref:\n")
        self.result_text.insert(tk.END, "=" * 40 + "\n")
        for doc in self.doctors:
            pref_str = " -> ".join(self.preferences[doc])
            self.result_text.insert(tk.END, f"{doc}: {pref_str}\n")
        self.result_text.insert(tk.END, "\n")

        # Show assignment results
        if self.assigned:
            self.result_text.insert(tk.END, "Assignment results (Based on Doctors):\n")
            self.result_text.insert(tk.END, "=" * 40 + "\n")
            for doc in self.doctors:
                hosp = self.assigned[doc]
                status = "Not assigned" if hosp is None else f"Assigned {hosp}"
                pref = self.preferences[doc]
                rank = "N/A" if hosp is None or hosp not in pref else str(pref.index(hosp) + 1)
                self.result_text.insert(tk.END, f"{doc}: {status} (Pref rank: {rank})\n")

        # Show results 2.0
            if self.assigned:
                self.result_text.insert(tk.END, "Assignment results (Based on Hospitals):\n")
                self.result_text.insert(tk.END, "=" * 40 + "\n")

            for hosp in self.hospitals:
                if hosp == "Not assigned":
                    self.result_text.insert(tk.END, f"\n Unassigned doctors:\n")
                else:
                    self.result_text.insert(tk.END, f"\n{hosp}:\n")
                self.result_text.insert(tk.END, "-" * 30 + "\n")

                for doc, assigned_hosp in self.assigned.items():
                    if (hosp == "Not assigned" and assigned_hosp is None) or assigned_hosp == hosp:
                        pref = self.preferences[doc]
                        rank = "N/A" if assigned_hosp is None or assigned_hosp not in pref else str(
                            pref.index(assigned_hosp) + 1)
                        status = "Not assigned" if assigned_hosp is None else f"Assigned"
                        self.result_text.insert(tk.END, f"  {doc} ({status}, Preference rank: {rank})\n")

            # Show cost matrix in detail area
            cost_matrix = self.prefs_to_cost(self.doctors, self.hospitals, self.preferences)
            self.detail_text.insert(tk.END, "Cost Matrix:\n")
            self.detail_text.insert(tk.END, "=" * 40 + "\n")
            self.detail_text.insert(tk.END, "Doc/Hos\t")
            for hosp in self.hospitals:
                self.detail_text.insert(tk.END, f"{hosp}\t")
            self.detail_text.insert(tk.END, "\n")

            for i, doc in enumerate(self.doctors):
                self.detail_text.insert(tk.END, f"{doc}\t\t")
                for j in range(len(self.hospitals)):
                    self.detail_text.insert(tk.END, f"{cost_matrix[i, j]:.1f}\t")
                self.detail_text.insert(tk.END, "\n")


def main():
    root = tk.Tk()
    app = HospitalDoctorAssignmentDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()