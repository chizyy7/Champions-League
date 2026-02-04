import tkinter as tk
import random
import pygame
import os

# ---------------- FILE PATHS ----------------
BASE_DIR = os.path.dirname(__file__)
WIN_SOUND = os.path.join(BASE_DIR, "win.wav")
LOSE_SOUND = os.path.join(BASE_DIR, "lose.wav")
BG_IMAGE = os.path.join(BASE_DIR, "background.png")
SCORE_FILE = os.path.join(BASE_DIR, "highscore.txt")

# ---------------- SOUND ----------------
pygame.mixer.init()
sound_on = True

def play_sound(sound):
    if sound_on:
        try:
            pygame.mixer.Sound(sound).play()
        except:
            pass

# ---------------- HIGH SCORE ----------------
def load_high_score():
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, "r") as f:
            return int(f.read())
    return 0

def save_high_score(score):
    with open(SCORE_FILE, "w") as f:
        f.write(str(score))

high_score = load_high_score()

# ---------------- WINDOW ----------------
window = tk.Tk()
window.title("🎮 Guessing Game")
window.geometry("420x560")
window.resizable(False, False)

# ---------------- BACKGROUND ----------------
bg_img = tk.PhotoImage(file=BG_IMAGE)
bg_label = tk.Label(window, image=bg_img)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# ---------------- GAME VARIABLES ----------------
secret = 0
lives = 0
score = 0
time_left = 0
timer_running = False

# ---------------- ANIMATIONS ----------------
def flash(label, c1, c2, count=6):
    if count == 0:
        return
    label.config(fg=c1 if count % 2 == 0 else c2)
    window.after(150, flash, label, c1, c2, count - 1)

def shake():
    x = window.winfo_x()
    y = window.winfo_y()
    for _ in range(10):
        window.geometry(f"+{x+5}+{y}")
        window.update()
        window.after(20)
        window.geometry(f"+{x-5}+{y}")
        window.update()
        window.after(20)

# ---------------- TIMER ----------------
def countdown():
    global time_left, timer_running
    if not timer_running:
        return
    if time_left > 0:
        time_left -= 1
        timer_label.config(text=f"⏱️ {time_left}s")
        window.after(1000, countdown)
    else:
        timer_running = False
        play_sound(LOSE_SOUND)
        flash(result_label, "red", "white")
        result_label.config(text=f"⏰ TIME UP!\nNumber was {secret}")

# ---------------- GAME LOGIC ----------------
def start_game(level):
    global secret, lives, time_left, timer_running
    timer_running = True

    if level == "easy":
        secret = random.randint(1, 10)
        lives = 5
        time_left = 30
    elif level == "medium":
        secret = random.randint(1, 20)
        lives = 4
        time_left = 25
    else:
        secret = random.randint(1, 50)
        lives = 3
        time_left = 20

    timer_label.config(text=f"⏱️ {time_left}s")
    result_label.config(text=f"Lives: {lives}", fg="white")
    countdown()

def check_guess():
    global lives, score, high_score, timer_running

    if not timer_running:
        return

    try:
        guess = int(entry.get())
    except:
        result_label.config(text="Enter a number!", fg="yellow")
        return

    if guess == secret:
        timer_running = False
        score += 1
        play_sound(WIN_SOUND)
        flash(result_label, "lime", "gold")
        result_label.config(text="🎉 YOU WON 🎉")

        if score > high_score:
            high_score = score
            save_high_score(high_score)
            high_score_label.config(text=f"🏆 High Score: {high_score}")

        score_label.config(text=f"Score: {score}")
    else:
        lives -= 1
        score -= 1
        shake()
        flash(result_label, "red", "white")
        score_label.config(text=f"Score: {score}")

        if lives > 0:
            hint = "Too low!" if guess < secret else "Too high!"
            result_label.config(text=f"{hint}\nLives: {lives}")
        else:
            timer_running = False
            play_sound(LOSE_SOUND)
            result_label.config(text=f"💀 GAME OVER\nNumber was {secret}")

def restart_game():
    global timer_running
    timer_running = False
    entry.delete(0, tk.END)
    result_label.config(text="Choose difficulty to start.")
    timer_label.config(text="⏱️ --")

def toggle_sound():
    global sound_on
    sound_on = not sound_on
    sound_btn.config(text="🔊 ON" if sound_on else "🔇 OFF")

# ---------------- UI ----------------
tk.Label(window, text="🎯 Guessing Game", font=("Arial", 18, "bold"),
         bg="#000", fg="white").pack(pady=10)

score_label = tk.Label(window, text="Score: 0", bg="#000", fg="white")
score_label.pack()

high_score_label = tk.Label(
    window,
    text=f"🏆 High Score: {high_score}",
    bg="#000",
    fg="gold"
)
high_score_label.pack()

timer_label = tk.Label(window, text="⏱️ --", bg="#000", fg="white")
timer_label.pack(pady=5)

entry = tk.Entry(window, font=("Arial", 14), justify="center")
entry.pack(pady=10)

tk.Button(window, text="Guess", command=check_guess).pack()

btn_frame = tk.Frame(window, bg="#000")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Easy", command=lambda: start_game("easy")).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Medium", command=lambda: start_game("medium")).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Hard", command=lambda: start_game("hard")).grid(row=0, column=2, padx=5)

sound_btn = tk.Button(window, text="🔊 ON", command=toggle_sound)
sound_btn.pack(pady=5)

tk.Button(window, text="Restart", command=restart_game).pack(pady=5)

result_label = tk.Label(window, text="Choose difficulty to start.",
                        font=("Arial", 12), bg="#000", fg="white")
result_label.pack(pady=15)

window.mainloop()