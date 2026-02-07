import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import threading
import os
import sys
import json
import queue
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Dark theme colors
COLORS = {
    'bg_dark': '#1a1d23',
    'bg_card': '#242830',
    'bg_input': '#2d323c',
    'text_primary': '#ffffff',
    'text_secondary': '#8b95a5',
    'accent_green': '#22c55e',
    'accent_blue': '#3b82f6',
    'accent_orange': '#f97316',
    'accent_cyan': '#06b6d4',
    'border': '#3d4450',
    'chart_bg': '#1e2128',
    'terminal_bg': '#0d1117',
    'terminal_fg': '#58a6ff',
}


class ModernButton(tk.Canvas):
    """Custom modern button with rounded corners"""
    def __init__(self, parent, text, command=None, bg_color=None, fg_color='white', 
                 width=120, height=36, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        
        self.command = command
        self.bg_color = bg_color or COLORS['accent_blue']
        self.fg_color = fg_color
        self.text = text
        self.enabled = True
        
        self.bind('<Button-1>', self._on_click)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        
        self._draw()
    
    def _draw(self, hover=False):
        self.delete('all')
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        
        color = self.bg_color if self.enabled else '#4a5568'
        if hover and self.enabled:
            color = self._lighten_color(color)
        
        r = 6
        self.create_oval(0, 0, r*2, r*2, fill=color, outline='')
        self.create_oval(w-r*2, 0, w, r*2, fill=color, outline='')
        self.create_oval(0, h-r*2, r*2, h, fill=color, outline='')
        self.create_oval(w-r*2, h-r*2, w, h, fill=color, outline='')
        self.create_rectangle(r, 0, w-r, h, fill=color, outline='')
        self.create_rectangle(0, r, w, h-r, fill=color, outline='')
        
        text_color = self.fg_color if self.enabled else '#9ca3af'
        self.create_text(w//2, h//2, text=self.text, fill=text_color, 
                        font=('Segoe UI', 10, 'bold'))
    
    def _lighten_color(self, color):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        r = min(255, r + 30)
        g = min(255, g + 30)
        b = min(255, b + 30)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _on_click(self, event):
        if self.enabled and self.command:
            self.command()
    
    def _on_enter(self, event):
        self._draw(hover=True)
    
    def _on_leave(self, event):
        self._draw(hover=False)
    
    def set_enabled(self, enabled):
        self.enabled = enabled
        self._draw()


class MetricCard(tk.Frame):
    """Modern metric card widget"""
    def __init__(self, parent, title, value, unit='', color=None, **kwargs):
        super().__init__(parent, bg=COLORS['bg_card'], **kwargs)
        
        self.color = color or COLORS['accent_green']
        self.config(highlightbackground=self.color, highlightthickness=2)
        
        self.title_label = tk.Label(self, text=title, 
                                    font=('Segoe UI', 10),
                                    fg=COLORS['text_secondary'],
                                    bg=COLORS['bg_card'])
        self.title_label.pack(anchor='w', padx=15, pady=(12, 0))
        
        value_frame = tk.Frame(self, bg=COLORS['bg_card'])
        value_frame.pack(anchor='w', padx=15, pady=(5, 12))
        
        self.value_label = tk.Label(value_frame, text=value,
                                    font=('Segoe UI', 24, 'bold'),
                                    fg=self.color,
                                    bg=COLORS['bg_card'])
        self.value_label.pack(side='left')
        
        if unit:
            self.unit_label = tk.Label(value_frame, text=f' {unit}',
                                       font=('Segoe UI', 12),
                                       fg=COLORS['text_secondary'],
                                       bg=COLORS['bg_card'])
            self.unit_label.pack(side='left', anchor='s', pady=(0, 4))
    
    def update_value(self, value):
        self.value_label.config(text=value)


class CollapsibleSection(tk.Frame):
    """Collapsible section with header"""
    def __init__(self, parent, title, icon='⚙', **kwargs):
        super().__init__(parent, bg=COLORS['bg_dark'], **kwargs)
        
        self.expanded = True
        
        header = tk.Frame(self, bg=COLORS['bg_dark'])
        header.pack(fill='x')
        
        self.toggle_btn = tk.Label(header, text='▼', 
                                   font=('Segoe UI', 10),
                                   fg=COLORS['text_secondary'],
                                   bg=COLORS['bg_dark'],
                                   cursor='hand2')
        self.toggle_btn.pack(side='right', padx=10)
        self.toggle_btn.bind('<Button-1>', self._toggle)
        
        icon_label = tk.Label(header, text=icon,
                             font=('Segoe UI', 12),
                             fg=COLORS['text_primary'],
                             bg=COLORS['bg_dark'])
        icon_label.pack(side='left', padx=(0, 8))
        
        title_label = tk.Label(header, text=title,
                              font=('Segoe UI', 11, 'bold'),
                              fg=COLORS['text_primary'],
                              bg=COLORS['bg_dark'])
        title_label.pack(side='left')
        
        self.content = tk.Frame(self, bg=COLORS['bg_dark'])
        self.content.pack(fill='x', pady=(10, 0))
    
    def _toggle(self, event=None):
        self.expanded = not self.expanded
        if self.expanded:
            self.content.pack(fill='x', pady=(10, 0))
            self.toggle_btn.config(text='▼')
        else:
            self.content.pack_forget()
            self.toggle_btn.config(text='▶')


class SettingRow(tk.Frame):
    """Setting row with label, slider, and value"""
    def __init__(self, parent, label, default_value, min_val=0, max_val=100, 
                 is_float=False, description='', **kwargs):
        super().__init__(parent, bg=COLORS['bg_dark'], **kwargs)
        
        self.is_float = is_float
        self.min_val = min_val
        self.max_val = max_val
        self.description = description
        self._popup = None
        
        lbl = tk.Label(self, text=label,
                      font=('Segoe UI', 10),
                      fg=COLORS['text_primary'],
                      bg=COLORS['bg_dark'],
                      cursor='hand2' if description else '')
        lbl.pack(anchor='w')
        
        if description:
            lbl.bind('<Button-1>', self._show_description)
        
        row = tk.Frame(self, bg=COLORS['bg_dark'])
        row.pack(fill='x', pady=(5, 0))
        
        self.var = tk.StringVar(value=str(default_value))
        self.entry = tk.Entry(row, textvariable=self.var,
                             font=('Segoe UI', 10),
                             bg=COLORS['bg_input'],
                             fg=COLORS['text_primary'],
                             insertbackground=COLORS['text_primary'],
                             relief='flat',
                             width=8)
        self.entry.pack(side='right', padx=(10, 0))
        
        self.slider = ttk.Scale(row, from_=min_val, to=max_val,
                               orient='horizontal',
                               command=self._on_slider_change)
        self.slider.set(float(default_value))
        self.slider.pack(side='left', fill='x', expand=True)
        
        self.entry.bind('<Return>', self._on_entry_change)
        self.entry.bind('<FocusOut>', self._on_entry_change)
    
    def _on_slider_change(self, value):
        val = float(value)
        if self.is_float:
            self.var.set(f'{val:.2f}')
        else:
            self.var.set(str(int(val)))
    
    def _on_entry_change(self, event=None):
        try:
            val = float(self.var.get())
            val = max(self.min_val, min(self.max_val, val))
            self.slider.set(val)
        except ValueError:
            pass
    
    def get(self):
        return self.var.get()
    
    def _show_description(self, event):
        if self._popup:
            self._popup.destroy()
            self._popup = None
            return
        
        self._popup = tk.Toplevel(self)
        self._popup.overrideredirect(True)
        self._popup.configure(bg=COLORS['border'])
        
        inner = tk.Frame(self._popup, bg=COLORS['bg_card'], padx=1, pady=1)
        inner.pack(fill='both', expand=True, padx=1, pady=1)
        
        msg = tk.Label(inner, text=self.description,
                      font=('Segoe UI', 9),
                      fg=COLORS['text_primary'],
                      bg=COLORS['bg_card'],
                      wraplength=280,
                      justify='right',
                      padx=12, pady=10)
        msg.pack()
        
        x = event.widget.winfo_rootx()
        y = event.widget.winfo_rooty() + event.widget.winfo_height() + 4
        self._popup.geometry(f'+{x}+{y}')
        
        self._popup.after(5000, lambda: self._close_popup())
        self._popup.bind('<Button-1>', lambda e: self._close_popup())
    
    def _close_popup(self):
        if self._popup:
            self._popup.destroy()
            self._popup = None


class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Meta-RL WSN Trainer")
        self.root.geometry("1200x800")
        self.root.minsize(1100, 750)
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Data storage for charts
        self.rounds = []
        self.energy_data = []
        self.delay_data = []
        self.reward_data = []
        self.connectivity_data = []
        
        # Training process
        self.training_process = None
        self.is_training = False
        self.output_queue = queue.Queue()
        
        # Current metrics
        self.current_reward = 0.0
        self.current_energy = 0.0
        self.current_delay = 0.0
        self.current_connectivity = 0.0
        self.initial_energy = None
        self.initial_delay = None
        
        self.setup_styles()
        self.setup_ui()
        self.setup_charts()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Dark.TFrame', background=COLORS['bg_dark'])
        style.configure('Card.TFrame', background=COLORS['bg_card'])
        
        style.configure('Dark.TLabel',
                       background=COLORS['bg_dark'],
                       foreground=COLORS['text_primary'],
                       font=('Segoe UI', 10))
        
        style.configure('Dark.TEntry',
                       fieldbackground=COLORS['bg_input'],
                       foreground=COLORS['text_primary'],
                       insertcolor=COLORS['text_primary'])
        
        style.configure('TScale',
                       background=COLORS['accent_blue'],
                       troughcolor=COLORS['bg_input'],
                       sliderrelief='flat',
                       sliderlength=20)
        
        style.map('TScale',
                 background=[('active', COLORS['accent_cyan'])])
        
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_frame.pack(fill='both', expand=True)
        
        self._create_header(main_frame)
        
        content = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True, padx=15, pady=10)
        
        self._create_sidebar(content)
        
        right_area = tk.Frame(content, bg=COLORS['bg_dark'])
        right_area.pack(side='right', fill='both', expand=True)
        
        self._create_metric_cards(right_area)
        self._create_charts_area(right_area)
        self._create_terminal(right_area)
    
    def _create_header(self, parent):
        header = tk.Frame(parent, bg=COLORS['bg_dark'], height=60)
        header.pack(fill='x', padx=15, pady=(10, 0))
        header.pack_propagate(False)
        
        left = tk.Frame(header, bg=COLORS['bg_dark'])
        left.pack(side='left', fill='y')
        
        title = tk.Label(left, text="Meta-RL WSN Trainer",
                        font=('Segoe UI', 18, 'bold'),
                        fg=COLORS['text_primary'],
                        bg=COLORS['bg_dark'])
        title.pack(side='left', anchor='w')
        
        subtitle = tk.Label(left, text="Nodes • Range • Steps • Device",
                           font=('Segoe UI', 9),
                           fg=COLORS['text_secondary'],
                           bg=COLORS['bg_dark'])
        subtitle.pack(side='left', padx=(15, 0), anchor='s', pady=(0, 5))
        
        right = tk.Frame(header, bg=COLORS['bg_dark'])
        right.pack(side='right', fill='y')
        
        status_frame = tk.Frame(right, bg=COLORS['bg_dark'])
        status_frame.pack(side='right', padx=(20, 0))
        
        self.status_dot = tk.Label(status_frame, text='●',
                                   font=('Segoe UI', 12),
                                   fg=COLORS['accent_green'],
                                   bg=COLORS['bg_dark'])
        self.status_dot.pack(side='left')
        
        self.status_label = tk.Label(status_frame, text='Idle',
                                    font=('Segoe UI', 10),
                                    fg=COLORS['text_secondary'],
                                    bg=COLORS['bg_dark'])
        self.status_label.pack(side='left', padx=(5, 0))
        
        self.btn_stop = ModernButton(right, "Stop Training", 
                                     command=self.stop_training,
                                     bg_color='#4a5568',
                                     width=110, height=34)
        self.btn_stop.pack(side='right', padx=5)
        self.btn_stop.set_enabled(False)
        
        self.btn_start = ModernButton(right, "Start Training",
                                      command=self.start_training,
                                      bg_color=COLORS['accent_blue'],
                                      width=110, height=34)
        self.btn_start.pack(side='right', padx=5)
    
    def _create_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=COLORS['bg_dark'], width=250)
        sidebar.pack(side='left', fill='y', padx=(0, 15))
        sidebar.pack_propagate(False)
        
        net_section = CollapsibleSection(sidebar, "Network Configuration", icon='⚙')
        net_section.pack(fill='x', pady=(0, 20))
        
        self.setting_nodes = SettingRow(net_section.content, "Number of Nodes", 
                                        100, 10, 500,
                                        description="Sensor count in the network.\nMore = bigger network, slower training.")
        self.setting_nodes.pack(fill='x', pady=5)
        
        self.setting_range = SettingRow(net_section.content, "Communication Range",
                                        0.15, 0.05, 0.5, is_float=True,
                                        description="Max distance nodes can talk.\nLow = weak links, harder task.\nHigh = strong links, easier task.")
        self.setting_range.pack(fill='x', pady=5)
        
        self.setting_energy = SettingRow(net_section.content, "Energy Consumption",
                                         0.05, 0.01, 0.2, is_float=True,
                                        description="Battery drain per step.\nHigh = batteries die fast,\nmodel must save energy better.")
        self.setting_energy.pack(fill='x', pady=5)
        
        self.setting_steps = SettingRow(net_section.content, "Max Steps",
                                        100, 10, 500,
                                        description="Steps per episode.\nMore = longer episodes, more data,\nbut slower training.")
        self.setting_steps.pack(fill='x', pady=5)
        
        train_section = CollapsibleSection(sidebar, "Training Settings", icon='⚙')
        train_section.pack(fill='x', pady=(0, 20))
        
        self.setting_iterations = SettingRow(train_section.content, "Meta Iterations",
                                             1000, 100, 10000,
                                        description="Total training rounds.\nMore = better model, takes longer.")
        self.setting_iterations.pack(fill='x', pady=5)
        
        self.setting_batch = SettingRow(train_section.content, "Meta Batch Size",
                                        5, 1, 20,
                                        description="Tasks per round.\nMore = stable learning, but slower.")
        self.setting_batch.pack(fill='x', pady=5)
        
        self.setting_adapt = SettingRow(train_section.content, "Adaptation Steps",
                                        5, 1, 20,
                                        description="Inner-loop updates per task.\nMore = deeper adaptation,\nbut slower per round.")
        self.setting_adapt.pack(fill='x', pady=5)
        
        ckpt_frame = tk.Frame(train_section.content, bg=COLORS['bg_dark'])
        ckpt_frame.pack(fill='x', pady=5)
        
        ckpt_label = tk.Label(ckpt_frame, text="Checkpoint Dir",
                font=('Segoe UI', 10),
                fg=COLORS['text_primary'],
                bg=COLORS['bg_dark'],
                cursor='hand2')
        ckpt_label.pack(anchor='w')
        self._ckpt_popup = None
        ckpt_label.bind('<Button-1>', self._show_ckpt_desc)
        
        self.entry_checkpoint = tk.Entry(ckpt_frame,
                                        font=('Segoe UI', 10),
                                        bg=COLORS['bg_input'],
                                        fg=COLORS['text_primary'],
                                        insertbackground=COLORS['text_primary'],
                                        relief='flat')
        self.entry_checkpoint.pack(fill='x', pady=(5, 0))
        self.entry_checkpoint.insert(0, "checkpoints")
    
    def _show_ckpt_desc(self, event):
        if self._ckpt_popup:
            self._ckpt_popup.destroy()
            self._ckpt_popup = None
            return
        self._ckpt_popup = tk.Toplevel(self.root)
        self._ckpt_popup.overrideredirect(True)
        self._ckpt_popup.configure(bg=COLORS['border'])
        inner = tk.Frame(self._ckpt_popup, bg=COLORS['bg_card'], padx=1, pady=1)
        inner.pack(fill='both', expand=True, padx=1, pady=1)
        msg = tk.Label(inner, text="Folder to save the best model file.",
                      font=('Segoe UI', 9),
                      fg=COLORS['text_primary'],
                      bg=COLORS['bg_card'],
                      wraplength=280, justify='left',
                      padx=12, pady=10)
        msg.pack()
        x = event.widget.winfo_rootx()
        y = event.widget.winfo_rooty() + event.widget.winfo_height() + 4
        self._ckpt_popup.geometry(f'+{x}+{y}')
        self._ckpt_popup.after(5000, lambda: self._close_ckpt_popup())
        self._ckpt_popup.bind('<Button-1>', lambda e: self._close_ckpt_popup())

    def _close_ckpt_popup(self):
        if self._ckpt_popup:
            self._ckpt_popup.destroy()
            self._ckpt_popup = None

    def _create_metric_cards(self, parent):
        cards_frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        cards_frame.pack(fill='x', pady=(0, 10))
        
        for i in range(4):
            cards_frame.columnconfigure(i, weight=1, uniform='cards')
        
        self.card_reward = MetricCard(cards_frame, "Reward", "+0.000",
                                      color=COLORS['accent_green'])
        self.card_reward.grid(row=0, column=0, sticky='ew', padx=(0, 8))
        
        self.card_energy = MetricCard(cards_frame, "Avg. Energy", "0.00000", "units",
                                      color=COLORS['accent_cyan'])
        self.card_energy.grid(row=0, column=1, sticky='ew', padx=8)
        
        self.card_delay = MetricCard(cards_frame, "Avg. Delay", "0.00", "ms",
                                     color=COLORS['accent_orange'])
        self.card_delay.grid(row=0, column=2, sticky='ew', padx=8)
        
        self.card_connectivity = MetricCard(cards_frame, "Connectivity", "0.0%",
                                            color=COLORS['text_primary'])
        self.card_connectivity.grid(row=0, column=3, sticky='ew', padx=(8, 0))
        
        # Energy reduction insight card
        insight_frame = tk.Frame(cards_frame, bg=COLORS['bg_card'])
        insight_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(12, 0))
        insight_frame.columnconfigure(0, weight=1)
        
        title = tk.Label(insight_frame, text="Energy Reduction vs Delay",
                         font=('Segoe UI', 11, 'bold'),
                         fg=COLORS['text_primary'],
                         bg=COLORS['bg_card'])
        title.pack(anchor='w')
        
        self.energy_reduction_label = tk.Label(
            insight_frame,
            text="Energy drop: 0.0%",
            font=('Segoe UI', 12, 'bold'),
            fg=COLORS['accent_green'],
            bg=COLORS['bg_card']
        )
        self.energy_reduction_label.pack(anchor='w', pady=(4, 0))
        
        self.delay_compare_label = tk.Label(
            insight_frame,
            text="Current delay: 0.00 ms",
            font=('Segoe UI', 10),
            fg=COLORS['text_secondary'],
            bg=COLORS['bg_card']
        )
        self.delay_compare_label.pack(anchor='w')
    
    def _create_charts_area(self, parent):
        self.charts_frame = tk.Frame(parent, bg=COLORS['bg_card'])
        self.charts_frame.pack(fill='both', expand=True, pady=(0, 10))
    
    def _create_terminal(self, parent):
        terminal_frame = tk.Frame(parent, bg=COLORS['bg_card'])
        terminal_frame.pack(fill='x')
        
        term_header = tk.Frame(terminal_frame, bg=COLORS['bg_card'])
        term_header.pack(fill='x', padx=10, pady=5)
        
        btn_frame = tk.Frame(term_header, bg=COLORS['bg_card'])
        btn_frame.pack(side='right')
        
        self.btn_save_logs = ModernButton(btn_frame, "Save Logs",
                                          command=self.save_logs,
                                          bg_color=COLORS['accent_green'],
                                          width=100, height=28)
        self.btn_save_logs.pack(side='right', padx=(5, 0))
        
        self.btn_copy_logs = ModernButton(btn_frame, "Copy Logs",
                                          command=self.copy_logs,
                                          bg_color=COLORS['bg_input'],
                                          width=100, height=28)
        self.btn_copy_logs.pack(side='right')
        
        self.terminal_output = scrolledtext.ScrolledText(
            terminal_frame, 
            height=8,
            bg=COLORS['terminal_bg'],
            fg=COLORS['terminal_fg'],
            font=('Consolas', 9),
            relief='flat',
            insertbackground=COLORS['terminal_fg']
        )
        self.terminal_output.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        self.terminal_output.tag_configure('green', foreground='#22c55e')
        self.terminal_output.tag_configure('orange', foreground='#f97316')
        self.terminal_output.tag_configure('cyan', foreground='#06b6d4')
        self.terminal_output.tag_configure('red', foreground='#ef4444')
        
    def setup_charts(self):
        self.fig = Figure(figsize=(10, 4), dpi=100, facecolor=COLORS['bg_card'])
        
        self.ax_energy = self.fig.add_subplot(121)
        self.ax_energy.set_facecolor(COLORS['chart_bg'])
        self.ax_energy.set_title("Energy Consumption vs Round", 
                                 fontsize=11, fontweight='bold', color=COLORS['text_primary'])
        self.ax_energy.set_xlabel("Round (Episode)", color=COLORS['text_secondary'])
        self.ax_energy.set_ylabel("Energy Consumption", color=COLORS['text_secondary'])
        self.ax_energy.tick_params(colors=COLORS['text_secondary'])
        self.ax_energy.grid(True, linestyle='--', alpha=0.3, color=COLORS['border'])
        for spine in self.ax_energy.spines.values():
            spine.set_color(COLORS['border'])
        self.line_energy, = self.ax_energy.plot([], [], color=COLORS['accent_cyan'], 
                                                 linewidth=2)
        
        self.ax_delay = self.fig.add_subplot(122)
        self.ax_delay.set_facecolor(COLORS['chart_bg'])
        self.ax_delay.set_title("Delay vs Round", 
                                fontsize=11, fontweight='bold', color=COLORS['text_primary'])
        self.ax_delay.set_xlabel("Round (Episode)", color=COLORS['text_secondary'])
        self.ax_delay.set_ylabel("Delay (ms)", color=COLORS['text_secondary'])
        self.ax_delay.tick_params(colors=COLORS['text_secondary'])
        self.ax_delay.grid(True, linestyle='--', alpha=0.3, color=COLORS['border'])
        for spine in self.ax_delay.spines.values():
            spine.set_color(COLORS['border'])
        self.line_delay, = self.ax_delay.plot([], [], color=COLORS['accent_orange'], 
                                               linewidth=1)
        
        self.fig.tight_layout(pad=2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.charts_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
    def update_charts(self, round_num, energy, delay):
        self.rounds.append(round_num)
        self.energy_data.append(energy)
        self.delay_data.append(delay)
        
        # Update Energy Chart
        self.line_energy.set_data(self.rounds, self.energy_data)
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        
        # Update Delay Chart
        self.line_delay.set_data(self.rounds, self.delay_data)
        self.ax_delay.relim()
        self.ax_delay.autoscale_view()
        
        self.canvas.draw()
        
    def clear_charts(self):
        self.rounds = []
        self.energy_data = []
        self.delay_data = []
        
        self.line_energy.set_data([], [])
        self.line_delay.set_data([], [])
        
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        self.ax_delay.relim()
        self.ax_delay.autoscale_view()
        
        self.canvas.draw()
        self.terminal_output.delete(1.0, tk.END)
        
        # Reset metric cards
        self.update_metric_cards(0.0, 0.0, 0.0, 0.0)
    
    def update_metric_cards(self, reward=None, energy=None, delay=None, connectivity=None):
        if reward is not None:
            self.current_reward = reward
            sign = '+' if reward >= 0 else ''
            self.card_reward.update_value(f"{sign}{reward:.3f}")
        
        if energy is not None:
            self.current_energy = energy
            self.card_energy.update_value(f"{energy:.5f}")
            if self.initial_energy is None and energy > 0:
                self.initial_energy = energy
        
        if delay is not None:
            self.current_delay = delay
            self.card_delay.update_value(f"{delay:.2f}")
            if self.initial_delay is None and delay > 0:
                self.initial_delay = delay
        
        if connectivity is not None:
            self.current_connectivity = connectivity
            self.card_connectivity.update_value(f"{connectivity:.1f}%")
        
        self._update_energy_delay_insight()

    def _update_energy_delay_insight(self):
        if self.initial_energy:
            change = ((self.initial_energy - self.current_energy) / self.initial_energy) * 100
            self.energy_reduction_label.config(
                text=f"Energy drop: {change:+.1f}%"
            )
            self.energy_reduction_label.config(
                fg=COLORS['accent_green'] if change >= 0 else COLORS['accent_orange']
            )
        else:
            self.energy_reduction_label.config(text="Energy drop: 0.0%")
        
        self.delay_compare_label.config(
            text=f"Current delay: {self.current_delay:.2f} ms"
        )
    
    def copy_logs(self):
        logs = self.terminal_output.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(logs)
        
    def save_logs(self):
        logs = self.terminal_output.get(1.0, tk.END)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(logs)
        
    def validate_inputs(self):
        try:
            num_nodes = int(self.setting_nodes.get())
            if num_nodes < 100:
                messagebox.showwarning("Warning", "Number of nodes should be 100 or more for optimal training.")
            
            comm_range = float(self.setting_range.get())
            energy_consumption = float(self.setting_energy.get())
            max_steps = int(self.setting_steps.get())
            meta_iterations = int(self.setting_iterations.get())
            meta_batch = int(self.setting_batch.get())
            adaptation_steps = int(self.setting_adapt.get())
            
            if num_nodes <= 0 or comm_range <= 0 or energy_consumption <= 0:
                raise ValueError("All values must be positive")
                
            return {
                'num_nodes': num_nodes,
                'comm_range': comm_range,
                'energy_consumption': energy_consumption,
                'max_steps': max_steps,
                'meta_iterations': meta_iterations,
                'meta_batch': meta_batch,
                'adaptation_steps': adaptation_steps,
                'checkpoint_dir': self.entry_checkpoint.get()
            }
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input values: {e}")
            return None
            
    def start_training(self):
        config = self.validate_inputs()
        if config is None:
            return
            
        self.is_training = True
        self.btn_start.set_enabled(False)
        self.btn_stop.set_enabled(True)
        self.status_label.config(text="Training...")
        self.status_dot.config(fg=COLORS['accent_orange'])
        self.clear_charts()
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.run_training, args=(config,), daemon=True)
        self.training_thread.start()
        
        # Start monitoring output
        self.root.after(100, self.check_output)
        
    def run_training(self, config):
        # Create training script with custom config
        script_path = os.path.join(os.path.dirname(__file__), "train_with_gui.py")
        
        # Write temporary training script
        training_code = f'''
import os
import sys
import traceback

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

print("Initializing training script...", flush=True)

try:
    import torch
    import numpy as np
    import time
    print("Libraries loaded successfully", flush=True)

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.envs.wsn_env import WSNEnv
    from src.networks.wsn_policy import WSNActorCritic
    from src.agents.maml_agent import MAMLAgent
    print("Modules imported successfully", flush=True)
except Exception as e:
    print(f"ERROR importing modules: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)

def create_tasks(num_tasks, env_config):
    tasks = []
    for _ in range(num_tasks):
        config = env_config.copy()
        base = env_config['comm_range']
        config['comm_range'] = np.random.uniform(
            max(0.05, base - 0.05),
            min(0.95, base + 0.05)
        )
        env = WSNEnv(config)
        tasks.append(env)
    return tasks

def collect_rollout(env, policy, max_steps=None):
    if max_steps is None:
        max_steps = float('inf')
    
    states, actions, rewards, dones = [], [], [], []
    state = env.reset()
    done = False
    step = 0
    total_energy = 0
    total_delay = 0
    
    while not done and step < max_steps:
        action = policy.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Calculate energy consumption (based on transmit power AND sleep schedule)
        active_mask = 1 - action['sleep_schedule']  # 1 if active, 0 if sleeping
        energy = env.energy_consumption * np.mean(active_mask * action['transmit_power'])
        total_energy += energy
        
        # Calculate delay (simulated based on network connectivity and distance)
        connectivity = np.sum(next_state['connectivity']) / (env.num_nodes * (env.num_nodes - 1))
        delay = (1 - connectivity) * 100 + np.random.uniform(0, 3)  # ms (reduced noise)
        total_delay += delay
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        state = next_state
        step += 1
    
    avg_energy = total_energy / max(step, 1)
    avg_delay = total_delay / max(step, 1)
    
    states_stacked = {{
        k: np.array([s[k] for s in states], dtype=np.float32)
        for k in states[0].keys()
    }}
    return {{
        'states': states_stacked,
        'actions': {{k: np.array([a[k] for a in actions], dtype=np.float32) for k in actions[0].keys()}},
        'rewards': np.array(rewards, dtype=np.float32),
        'dones': np.array(dones, dtype=bool),
        'avg_energy': avg_energy,
        'avg_delay': avg_delay
    }}

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration from GUI
    env_config = {{
        'num_nodes': {config['num_nodes']},
        'comm_range': {config['comm_range']},
        'energy_consumption': {config['energy_consumption']},
        'max_steps': {config['max_steps']}
    }}
    
    num_meta_iterations = {config['meta_iterations']}
    meta_batch_size = {config['meta_batch']}
    num_adaptation_steps = {config['adaptation_steps']}
    save_dir = '{config['checkpoint_dir']}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training with {{env_config['num_nodes']}} nodes on {{device}}", flush=True)
    print(f"Meta iterations: {{num_meta_iterations}}, Batch size: {{meta_batch_size}}", flush=True)
    print("=" * 60, flush=True)
    
    sample_env = WSNEnv(env_config)
    obs_space = sample_env.observation_space
    act_space = sample_env.action_space
    
    state_dim = (
        obs_space['node_positions'].shape[0] * obs_space['node_positions'].shape[1] +
        obs_space['battery_levels'].shape[0] +
        (obs_space['connectivity'].shape[0] * (obs_space['connectivity'].shape[0] - 1)) // 2
    )
    
    action_dims = {{
        'transmit_power': act_space['transmit_power'].shape[0],
        'sleep_schedule': act_space['sleep_schedule'].n
    }}
    
    policy = WSNActorCritic(state_dim, action_dims).to(device)
    
    agent = MAMLAgent(
        policy_network=policy,
        inner_lr=1e-3,
        meta_lr=1e-4,
        num_updates=num_adaptation_steps,
        device=device
    )
    
    best_avg_reward = -float('inf')
    
    for meta_iter in range(num_meta_iterations):
        tasks = create_tasks(meta_batch_size, env_config)
        
        task_data = []
        total_energy = 0
        total_delay = 0
        
        for task in tasks:
            rollout = collect_rollout(task, policy, env_config['max_steps'])
            task_data.append(rollout)
            total_energy += rollout['avg_energy']
            total_delay += rollout['avg_delay']
        
        avg_energy = total_energy / meta_batch_size
        avg_delay = total_delay / meta_batch_size
        
        meta_loss = agent.meta_update(task_data)
        
        # Calculate connectivity
        eval_env = WSNEnv(env_config)
        eval_state = eval_env.reset()
        connectivity = np.sum(eval_state['connectivity']) / (env_config['num_nodes'] * (env_config['num_nodes'] - 1)) * 100
        
        # Print metrics for GUI parsing
        progress = ((meta_iter + 1) / num_meta_iterations) * 100
        print(f"METRICS|{{meta_iter + 1}}|{{avg_energy:.6f}}|{{avg_delay:.2f}}|{{progress:.1f}}|{{connectivity:.1f}}", flush=True)
        
        if (meta_iter + 1) % 10 == 0:
            eval_rollout = collect_rollout(eval_env, policy, env_config['max_steps'])
            avg_reward = np.mean(eval_rollout['rewards'])
            
            print(f"Round {{meta_iter + 1}}: Loss={{meta_loss:.6f}}, Reward={{avg_reward:.3f}}, Energy={{avg_energy:.6f}}, Delay={{avg_delay:.2f}}ms", flush=True)
            print(f"REWARD|{{avg_reward:.3f}}", flush=True)
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                save_path = os.path.join(save_dir, 'best_model.pt')
                agent.save(save_path)
                print(f"New best model saved! Reward: {{avg_reward:.3f}}", flush=True)
    
    print("=" * 60, flush=True)
    print("=== Training finished ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main: {{e}}", flush=True)
        traceback.print_exc()
        sys.exit(1)

'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(training_code)
        
        try:
            # Run training script with unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            self.training_process = subprocess.Popen(
                [sys.executable, '-u', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(__file__),
                env=env
            )
            
            # Read output line by line
            for line in iter(self.training_process.stdout.readline, ''):
                if not self.is_training:
                    break
                self.output_queue.put(line)
                
            self.training_process.wait()
            
        except Exception as e:
            self.output_queue.put(f"Error: {str(e)}\n")
        finally:
            self.output_queue.put("TRAINING_DONE")
            
    def check_output(self):
        try:
            while True:
                line = self.output_queue.get_nowait()
                
                if line == "TRAINING_DONE":
                    self.training_finished()
                    return
                    
                # Parse metrics line
                if line.startswith("METRICS|"):
                    parts = line.strip().split("|")
                    if len(parts) >= 5:
                        round_num = int(parts[1])
                        energy = float(parts[2])
                        delay = float(parts[3])
                        progress = float(parts[4])
                        connectivity = float(parts[5]) if len(parts) > 5 else 0
                        
                        self.update_charts(round_num, energy, delay)
                        self.update_metric_cards(energy=energy, delay=delay, 
                                                connectivity=connectivity)
                elif line.startswith("REWARD|"):
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        reward = float(parts[1])
                        self.update_metric_cards(reward=reward)
                else:
                    # Display in terminal with colors
                    if "Error" in line or "ERROR" in line:
                        self.terminal_output.insert(tk.END, line, 'red')
                    elif "best model" in line.lower():
                        self.terminal_output.insert(tk.END, line, 'green')
                    elif "Round" in line:
                        self._insert_colored_round(line)
                    elif "finished" in line.lower():
                        self.terminal_output.insert(tk.END, line, 'green')
                    else:
                        self.terminal_output.insert(tk.END, line)
                    self.terminal_output.see(tk.END)
                    
        except queue.Empty:
            pass
            
        if self.is_training:
            self.root.after(100, self.check_output)
            
    def _insert_colored_round(self, line):
        """Insert a round line with colored metrics"""
        parts = line.split(', ')
        for i, part in enumerate(parts):
            if 'Loss=' in part:
                self.terminal_output.insert(tk.END, part)
            elif 'Reward=' in part:
                self.terminal_output.insert(tk.END, part, 'green')
            elif 'Energy=' in part:
                self.terminal_output.insert(tk.END, part, 'cyan')
            elif 'Delay=' in part:
                self.terminal_output.insert(tk.END, part, 'orange')
            else:
                self.terminal_output.insert(tk.END, part)
            
            if i < len(parts) - 1:
                self.terminal_output.insert(tk.END, ', ')
        self.terminal_output.insert(tk.END, '\n')
    
    def training_finished(self):
        self.is_training = False
        self.btn_start.set_enabled(True)
        self.btn_stop.set_enabled(False)
        self.status_label.config(text="Idle")
        self.status_dot.config(fg=COLORS['accent_green'])
        self.terminal_output.insert(tk.END, "\n=== Training Finished ===\n", 'green')
        self.terminal_output.see(tk.END)
        
    def stop_training(self):
        self.is_training = False
        if self.training_process:
            self.training_process.terminate()
            self.training_process = None
            
        self.btn_start.set_enabled(True)
        self.btn_stop.set_enabled(False)
        self.status_label.config(text="Stopped")
        self.status_dot.config(fg='#ef4444')
        self.terminal_output.insert(tk.END, "\n=== Training Stopped by User ===\n", 'red')
        self.terminal_output.see(tk.END)


def main():
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
