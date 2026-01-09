import tkinter as tk
from tkinter import messagebox

import torch
import numpy as np

from test_agent import evaluate_agent


# ==========================================
# 2. منطق تشغيل التقييم (Evaluation Logic)
# يربط بين واجهة المستخدم ودالة الاختبار في test_agent.py
# ==========================================
def run_evaluation():
    try:
        num_nodes = int(entry_num_nodes.get())
        comm_range = float(entry_comm_range.get())
        num_episodes = int(entry_num_episodes.get())
        checkpoint_path = entry_checkpoint.get().strip()

        if num_nodes <= 0:
            raise ValueError("عدد الحساسات يجب أن يكون أكبر من صفر")
        if num_episodes <= 0:
            raise ValueError("عدد الحلقات يجب أن يكون أكبر من صفر")
        if comm_range <= 0:
            raise ValueError("مدى الاتصال يجب أن يكون أكبر من صفر")

        env_config = {
            'num_nodes': num_nodes,
            'comm_range': comm_range,
            'energy_consumption': 0.05,
            'max_steps': 100,
        }

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rewards = evaluate_agent(
            checkpoint_path=checkpoint_path,
            num_episodes=num_episodes,
            device=device,
            env_config=env_config,
        )

        avg = float(np.mean(rewards))
        std = float(np.std(rewards))

        msg = (
            f"تم التقييم بنجاح!\n\n"
            f"متوسط العائد (reward) على {num_episodes} حلقة: {avg:.3f}\n"
            f"الانحراف المعياري: {std:.3f}"
        )
        messagebox.showinfo("نتيجة التقييم", msg)

    except FileNotFoundError as e:
        messagebox.showerror("خطأ في الملف", f"لم يتم العثور على ملف النموذج:\n{e}")
    except ValueError as e:
        messagebox.showerror("خطأ في المدخلات", str(e))
    except Exception as e:
        messagebox.showerror("خطأ غير متوقع", f"حدث خطأ غير متوقع:\n{e}")


# ==========================================
# 1. تهيئة الواجهة الرسومية (GUI Setup)
# ==========================================
root = tk.Tk()
root.title("تقييم نموذج التحكم في شبكة الحساسات")

default_font = ("Tahoma", 10)
root.option_add("*Font", default_font)

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill=tk.BOTH, expand=True)

label_num_nodes = tk.Label(frame, text="عدد الحساسات (num_nodes):")
label_num_nodes.grid(row=0, column=0, sticky="w", pady=5)
entry_num_nodes = tk.Entry(frame)
entry_num_nodes.grid(row=0, column=1, pady=5)
entry_num_nodes.insert(0, "10")

label_comm_range = tk.Label(frame, text="مدى الاتصال بين العقد (comm_range):")
label_comm_range.grid(row=1, column=0, sticky="w", pady=5)
entry_comm_range = tk.Entry(frame)
entry_comm_range.grid(row=1, column=1, pady=5)
entry_comm_range.insert(0, "0.3")

label_num_episodes = tk.Label(frame, text="عدد الحلقات (episodes):")
label_num_episodes.grid(row=2, column=0, sticky="w", pady=5)
entry_num_episodes = tk.Entry(frame)
entry_num_episodes.grid(row=2, column=1, pady=5)
entry_num_episodes.insert(0, "10")

label_checkpoint = tk.Label(frame, text="مسار ملف النموذج (checkpoint):")
label_checkpoint.grid(row=3, column=0, sticky="w", pady=5)
entry_checkpoint = tk.Entry(frame, width=40)
entry_checkpoint.grid(row=3, column=1, pady=5)
entry_checkpoint.insert(0, "checkpoints/best_model.pt")

btn_run = tk.Button(frame, text="تشغيل التقييم", command=run_evaluation)
btn_run.grid(row=4, column=0, columnspan=2, pady=15)

label_note = tk.Label(
    frame,
    text=(
        "ملاحظة: النموذج المدرَّب في الملف الافتراضي مهيأ لـ 10 حساسات. "
        "إذا أدخلت عدد حساسات مختلفًا فسيتم استخدام نموذج غير مدرَّب (أوزان عشوائية)، "
        "وسيتم أيضًا عرض منحنى العوائد (rewards) لكل حلقة في نافذة رسم منفصلة."
    ),
    fg="gray",
    wraplength=500,
    justify="right",
)
label_note.grid(row=5, column=0, columnspan=2, sticky="w")


if __name__ == "__main__":
    root.mainloop()
