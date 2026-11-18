import sys
import threading
from datetime import date
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from smartface import database
from smartface.face_engine import FaceEngine

ROOT = Path(__file__).resolve().parent

class SmartFaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartFace Attendance System")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)
        self.root.configure(bg="#f3f5f9")
        database.initialize()
        self.engine = None
        self.camera = None
        self.camera_mode = None
        self.running = False
        self.photo = None
        self.registration_features = []
        self.frame_counter = 0
        self.last_recognition = {}
        self.last_marked = None
        self._build_ui()
        self.refresh_all()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.set_status("Ready. Load the face engine to begin.")

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background="#f3f5f9", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        header = tk.Frame(self.root, bg="#172554", height=76)
        header.pack(fill="x")
        tk.Label(header, text="SMARTFACE", fg="white", bg="#172554",
                 font=("Segoe UI", 20, "bold")).pack(side="left", padx=24, pady=18)
        tk.Label(header, text="FACE ATTENDANCE MANAGEMENT", fg="#bfdbfe", bg="#172554",
                 font=("Segoe UI", 10)).pack(side="left", pady=25)
        self.status_var = tk.StringVar()
        tk.Label(header, textvariable=self.status_var, fg="#dbeafe", bg="#172554",
                 font=("Segoe UI", 9)).pack(side="right", padx=20)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=18, pady=16)
        self.dashboard_tab = ttk.Frame(self.tabs, padding=16)
        self.register_tab = ttk.Frame(self.tabs, padding=16)
        self.recognition_tab = ttk.Frame(self.tabs, padding=16)
        self.employees_tab = ttk.Frame(self.tabs, padding=16)
        self.reports_tab = ttk.Frame(self.tabs, padding=16)
        for tab, title in [
            (self.dashboard_tab, "Dashboard"), (self.register_tab, "Register Face"),
            (self.recognition_tab, "Take Attendance"), (self.employees_tab, "Employees"),
            (self.reports_tab, "Reports")
        ]:
            self.tabs.add(tab, text=title)
        self._build_dashboard()
        self._build_registration()
        self._build_recognition()
        self._build_employees()
        self._build_reports()

    def set_status(self, message):
        self.status_var.set(message)

    def load_engine(self):
        if self.engine is not None:
            return True
        self.set_status("Loading face models...")
        self.root.update_idletasks()
        try:
            self.engine = FaceEngine()
            self.set_status("Face engine loaded.")
            return True
        except Exception as exc:
            messagebox.showerror("Model setup failed",
                f"Could not load face models.\n\n{exc}\n\nCheck your internet connection and try again.")
            self.set_status("Face engine unavailable.")
            return False

    def _build_dashboard(self):
        ttk.Label(self.dashboard_tab, text="Attendance overview",
                  font=("Segoe UI", 19, "bold")).pack(anchor="w", pady=(0, 14))
        cards = ttk.Frame(self.dashboard_tab)
        cards.pack(fill="x")
        self.employee_count = tk.StringVar(value="0")
        self.today_count = tk.StringVar(value="0")
        self._stat_card(cards, "REGISTERED EMPLOYEES", self.employee_count, 0)
        self._stat_card(cards, "PRESENT TODAY", self.today_count, 1)
        ttk.Label(self.dashboard_tab, text="Today's attendance",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(24, 8))
        self.dashboard_tree = self._tree(self.dashboard_tab,
            ("employee_id", "name", "attendance_date", "check_in"),
            ("Employee ID", "Name", "Date", "Check-in"))
        ttk.Button(self.dashboard_tab, text="Refresh", command=self.refresh_all).pack(anchor="e", pady=8)

    def _stat_card(self, parent, title, variable, column):
        card = tk.Frame(parent, bg="white", highlightbackground="#dbe2ee", highlightthickness=1)
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 12), ipadx=28, ipady=18)
        parent.columnconfigure(column, weight=1)
        tk.Label(card, text=title, bg="white", fg="#64748b",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(card, textvariable=variable, bg="white", fg="#172554",
                 font=("Segoe UI", 27, "bold")).pack(anchor="w", pady=(8, 0))

    def _build_registration(self):
        ttk.Label(self.register_tab, text="Register a new person",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w")
        form = ttk.Frame(self.register_tab)
        form.pack(fill="x", pady=12)
        ttk.Label(form, text="Employee ID").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.reg_id = tk.StringVar()
        ttk.Entry(form, textvariable=self.reg_id, width=25).grid(row=1, column=0, sticky="w", padx=(0, 18), pady=5)
        ttk.Label(form, text="Full name").grid(row=0, column=1, sticky="w")
        self.reg_name = tk.StringVar()
        ttk.Entry(form, textvariable=self.reg_name, width=32).grid(row=1, column=1, sticky="w", pady=5)
        ttk.Button(form, text="Start face capture (10 samples)", command=self.start_registration).grid(row=1, column=2, padx=18)
        ttk.Button(form, text="Stop camera", command=self.stop_camera).grid(row=1, column=3)
        self.reg_progress = tk.StringVar(value="Enter an ID and name, then start capture.")
        ttk.Label(self.register_tab, textvariable=self.reg_progress).pack(anchor="w", pady=4)
        self.reg_preview = tk.Label(self.register_tab, bg="#111827", text="Camera preview",
                                    fg="white", font=("Segoe UI", 12), width=90, height=25)
        self.reg_preview.pack(fill="both", expand=True, pady=8)
        ttk.Label(self.register_tab,
            text="Capture guidance: face the camera, remove obstructions, and slowly vary your head angle.").pack(anchor="w")

    def _build_recognition(self):
        ttk.Label(self.recognition_tab, text="Live attendance recognition",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w")
        controls = ttk.Frame(self.recognition_tab)
        controls.pack(fill="x", pady=12)
        ttk.Button(controls, text="Start attendance camera", command=self.start_recognition).pack(side="left")
        ttk.Button(controls, text="Stop camera", command=self.stop_camera).pack(side="left", padx=8)
        self.rec_status = tk.StringVar(value="Camera stopped.")
        ttk.Label(controls, textvariable=self.rec_status).pack(side="left", padx=16)
        self.rec_preview = tk.Label(self.recognition_tab, bg="#111827", text="Camera preview",
                                    fg="white", font=("Segoe UI", 12), width=90, height=27)
        self.rec_preview.pack(fill="both", expand=True, pady=8)
        ttk.Label(self.recognition_tab,
            text="Attendance is recorded once per employee per calendar day. Unknown faces are not recorded.").pack(anchor="w")

    def _build_employees(self):
        ttk.Label(self.employees_tab, text="Registered employees",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w", pady=(0, 10))
        self.employee_tree = self._tree(self.employees_tab,
            ("employee_id", "name", "created_at"), ("Employee ID", "Name", "Registered At"))
        controls = ttk.Frame(self.employees_tab)
        controls.pack(fill="x", pady=10)
        ttk.Button(controls, text="Refresh", command=self.refresh_all).pack(side="left")
        ttk.Button(controls, text="Delete selected employee", command=self.delete_selected).pack(side="left", padx=8)
        ttk.Label(self.employees_tab, text="Deleting an employee also deletes their attendance history.",
                  foreground="#9a3412").pack(anchor="w")

    def _build_reports(self):
        ttk.Label(self.reports_tab, text="Attendance reports",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w", pady=(0, 10))
        row = ttk.Frame(self.reports_tab)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="Date (YYYY-MM-DD):").pack(side="left")
        self.report_date = tk.StringVar(value=date.today().isoformat())
        ttk.Entry(row, textvariable=self.report_date, width=16).pack(side="left", padx=8)
        ttk.Button(row, text="Load report", command=self.load_report).pack(side="left")
        ttk.Button(row, text="Export CSV", command=self.export_csv).pack(side="left", padx=8)
        self.report_tree = self._tree(self.reports_tab,
            ("employee_id", "name", "attendance_date", "check_in", "method"),
            ("Employee ID", "Name", "Date", "Check-in", "Method"))
        self.load_report()

    def _tree(self, parent, columns, headings):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col, heading in zip(columns, headings):
            tree.heading(col, text=heading)
            tree.column(col, width=150, anchor="w")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        return tree

    def _fill_tree(self, tree, rows, columns):
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", "end", values=tuple(row.get(c, "") for c in columns))

    def refresh_all(self):
        employees, present = database.dashboard_counts()
        self.employee_count.set(str(employees))
        self.today_count.set(str(present))
        self._fill_tree(self.dashboard_tree, database.get_attendance(),
                        ("employee_id", "name", "attendance_date", "check_in"))
        self._fill_tree(self.employee_tree, database.get_employees(),
                        ("employee_id", "name", "created_at"))
        self.load_report()

    def start_camera(self, mode):
        if self.running:
            self.stop_camera()
        if not self.load_engine():
            return
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Camera error", "Could not open webcam. Close other camera apps and check camera permissions.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera = cap
        self.camera_mode = mode
        self.running = True
        self.frame_counter = 0
        self._camera_tick()

    def start_registration(self):
        employee_id, name = self.reg_id.get().strip(), self.reg_name.get().strip()
        if not employee_id or not name:
            messagebox.showwarning("Missing details", "Enter both Employee ID and full name.")
            return
        if any(e["employee_id"].lower() == employee_id.lower() for e in database.get_employees()):
            messagebox.showwarning("Duplicate ID", "That employee ID is already registered.")
            return
        self.registration_features = []
        self.reg_progress.set("Capturing: keep your face visible and slowly change angle.")
        self.start_camera("register")

    def start_recognition(self):
        if not database.get_employees():
            messagebox.showinfo("No employees", "Register at least one person before starting attendance.")
            return
        self.rec_status.set("Recognizing... please look toward the camera.")
        self.start_camera("recognize")

    def stop_camera(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.camera_mode = None
        self.set_status("Camera stopped.")
        if hasattr(self, "rec_status"):
            self.rec_status.set("Camera stopped.")

    def _camera_tick(self):
        if not self.running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Camera error", "Webcam stopped returning frames.")
            return
        self.frame_counter += 1
        try:
            faces = self.engine.extract(frame)
        except Exception as exc:
            self.stop_camera()
            messagebox.showerror("Recognition error", str(exc))
            return

        if self.camera_mode == "register":
            self._process_registration(frame, faces)
            preview = self.reg_preview
        else:
            self._process_recognition(frame, faces)
            preview = self.rec_preview

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((850, 500))
        self.photo = ImageTk.PhotoImage(image)
        preview.configure(image=self.photo, text="")
        self.root.after(25, self._camera_tick)

    def _process_registration(self, frame, faces):
        for feature, (x, y, w, h), confidence in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 120), 2)
            cv2.putText(frame, "Registration capture", (x, max(20, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)
        if faces and self.frame_counter % 8 == 0 and len(self.registration_features) < 10:
            # Use the largest detected face to reduce accidental multi-person enrollment.
            feature = max(faces, key=lambda f: f[1][2] * f[1][3])[0]
            self.registration_features.append(feature.copy())
            self.reg_progress.set(f"Captured {len(self.registration_features)}/10 samples.")
            if len(self.registration_features) >= 10:
                vectors = np.vstack(self.registration_features)
                average = vectors.mean(axis=0)
                average /= max(np.linalg.norm(average), 1e-12)
                try:
                    database.add_employee(self.reg_id.get().strip(), self.reg_name.get().strip(), average)
                    self.stop_camera()
                    self.reg_progress.set("Registration successful. You can register another person.")
                    self.refresh_all()
                    messagebox.showinfo("Registration complete",
                        f"{self.reg_name.get().strip()} has been registered successfully.")
                except Exception as exc:
                    self.stop_camera()
                    messagebox.showerror("Registration failed", str(exc))

    def _process_recognition(self, frame, faces):
        employees = database.get_employees()
        for feature, (x, y, w, h), confidence in faces:
            person, score = self.engine.match(feature, employees)
            if person:
                color = (0, 200, 80)
                label = f'{person["name"]} ({score:.2f})'
                if self.frame_counter % 12 == 0:
                    inserted = database.mark_attendance(person["employee_id"])
                    if inserted:
                        self.last_marked = person["name"]
                        self.rec_status.set(f'Attendance marked: {person["name"]}')
                        self.refresh_all()
                    else:
                        self.rec_status.set(f'Already marked today: {person["name"]}')
            else:
                color = (0, 0, 255)
                label = "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, max(20, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def delete_selected(self):
        selected = self.employee_tree.selection()
        if not selected:
            messagebox.showinfo("Select employee", "Select an employee first.")
            return
        values = self.employee_tree.item(selected[0], "values")
        employee_id, name = values[0], values[1]
        if messagebox.askyesno("Confirm deletion", f"Delete {name} ({employee_id}) and their attendance history?"):
            database.delete_employee(employee_id)
            self.refresh_all()

    def load_report(self):
        try:
            day = self.report_date.get().strip()
            date.fromisoformat(day)
            rows = database.get_attendance(day)
            self._fill_tree(self.report_tree, rows,
                ("employee_id", "name", "attendance_date", "check_in", "method"))
        except ValueError:
            messagebox.showwarning("Invalid date", "Enter the date as YYYY-MM-DD.")

    def export_csv(self):
        try:
            day = self.report_date.get().strip()
            date.fromisoformat(day)
        except ValueError:
            messagebox.showwarning("Invalid date", "Enter the date as YYYY-MM-DD.")
            return
        rows = database.get_attendance(day)
        if not rows:
            messagebox.showinfo("No records", "There are no attendance records for this date.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=f"attendance_{day}.csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if path:
            pd.DataFrame(rows).to_csv(path, index=False)
            messagebox.showinfo("Export complete", f"Report saved to:\n{path}")

    def on_close(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartFaceApp(root)
    root.mainloop()
