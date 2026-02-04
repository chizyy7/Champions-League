from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
import random


class StartScreen(FloatLayout):
    def __init__(self, start_callback, **kwargs):
        super().__init__(**kwargs)
        self.start_callback = start_callback
        self.selected_difficulty = None

        # Background
        self.add_widget(Image(
            source="assets/images/background.png",
            allow_stretch=True,
            keep_ratio=False
        ))

        # Title
        self.title = Label(
            text="Penalty Shoot Out",
            font_size=40,
            pos_hint={"center_x": 0.5, "top": 0.95}
        )
        self.add_widget(self.title)

        # Message
        self.message = Label(
            text="Choose difficulty",
            font_size=20,
            pos_hint={"center_x": 0.5, "y": 0.55}
        )
        self.add_widget(self.message)

        # Difficulty buttons
        self.buttons = {}
        levels = ["EASY", "MEDIUM", "HARD"]
        x_positions = [0.3, 0.5, 0.7]

        for level, x in zip(levels, x_positions):
            btn = Button(
                text=level,
                size_hint=(0.2, 0.1),
                pos_hint={"center_x": x, "y": 0.45}
            )
            btn.bind(on_press=self.select_difficulty)
            self.buttons[level] = btn
            self.add_widget(btn)

        # Kick off button
        self.kickoff = Button(
            text="KICK OFF",
            size_hint=(0.4, 0.12),
            pos_hint={"center_x": 0.5, "y": 0.25}
        )
        self.kickoff.bind(on_press=self.start_game)
        self.add_widget(self.kickoff)

    def select_difficulty(self, instance):
        self.selected_difficulty = instance.text.lower()
        for btn in self.buttons.values():
            btn.background_color = (1, 1, 1, 1)
        instance.background_color = (0, 1, 0, 1)
        self.message.text = f"{instance.text} selected"

    def start_game(self, instance):
        if not self.selected_difficulty:
            self.message.text = "Choose difficulty first!"
            return
        self.start_callback(self.selected_difficulty)


class GameScreen(FloatLayout):
    def __init__(self, difficulty, **kwargs):
        super().__init__(**kwargs)
        self.difficulty = difficulty
        self.score = 0
        self.shots = 0

        # Sounds
        self.goal_sound = SoundLoader.load("assets/sounds/goal.wav")
        self.save_sound = SoundLoader.load("assets/sounds/save.wav")

        # Background
        self.add_widget(Image(
            source="assets/images/background.png",
            allow_stretch=True,
            keep_ratio=False
        ))

        # Goal
        self.goal = Image(
            source="assets/images/goal.png",
            size_hint=(0.9, 0.3),
            pos_hint={"center_x": 0.5, "top": 0.85}
        )
        self.add_widget(self.goal)

        # Keeper
        self.keeper = Image(
            source="assets/images/keeper.png",
            size_hint=(0.2, 0.2),
            pos_hint={"center_x": 0.5, "y": 0.55}
        )
        self.add_widget(self.keeper)

        # Ball
        self.ball = Image(
            source="assets/images/ball.png",
            size_hint=(0.12, 0.12),
            pos_hint={"center_x": 0.5, "y": 0.1}
        )
        self.add_widget(self.ball)

        # Score label (TOP LEFT)
        self.score_label = Label(
            text="Score: 0",
            font_size=20,
            pos_hint={"x": 0.02, "top": 0.98}
        )
        self.add_widget(self.score_label)

        # Shoot buttons
        self.create_shoot_buttons()

    def create_shoot_buttons(self):
        positions = {
            "LEFT": 0.3,
            "CENTER": 0.5,
            "RIGHT": 0.7
        }
        for direction, x in positions.items():
            btn = Button(
                text=direction,
                size_hint=(0.2, 0.1),
                pos_hint={"center_x": x, "y": 0.0}
            )
            btn.bind(on_press=lambda inst, d=direction: self.shoot(d))
            self.add_widget(btn)

    def shoot(self, direction):
        self.shots += 1

        ball_targets = {
            "LEFT": 0.35,
            "CENTER": 0.5,
            "RIGHT": 0.65
        }

        keeper_targets = self.get_keeper_target(direction)

        ball_anim = Animation(
            pos_hint={"center_x": ball_targets[direction], "y": 0.6},
            duration=0.5
        )
        keeper_anim = Animation(
            pos_hint={"center_x": keeper_targets, "y": 0.55},
            duration=self.get_keeper_speed()
        )

        ball_anim.start(self.ball)
        keeper_anim.start(self.keeper)

        Clock.schedule_once(
            lambda dt: self.check_goal(direction, keeper_targets),
            0.6
        )

    def get_keeper_speed(self):
        return {
            "easy": 0.6,
            "medium": 0.4,
            "hard": 0.25
        }[self.difficulty]

    def get_keeper_target(self, direction):
        chance = random.random()
        if self.difficulty == "easy":
            return random.choice([0.35, 0.5, 0.65])
        if self.difficulty == "medium":
            return 0.5 if chance < 0.5 else random.choice([0.35, 0.65])
        return {"LEFT": 0.35, "CENTER": 0.5, "RIGHT": 0.65}[direction]

    def check_goal(self, direction, keeper_x):
        ball_x = {"LEFT": 0.35, "CENTER": 0.5, "RIGHT": 0.65}[direction]

        if abs(ball_x - keeper_x) < 0.05:
            if self.save_sound:
                self.save_sound.play()
        else:
            self.score += 1
            if self.goal_sound:
                self.goal_sound.play()

        self.score_label.text = f"Score: {self.score}"
        self.reset_positions()

    def reset_positions(self):
        self.ball.pos_hint = {"center_x": 0.5, "y": 0.1}
        self.keeper.pos_hint = {"center_x": 0.5, "y": 0.55}


class PenaltyApp(App):
    def build(self):
        self.root = FloatLayout()
        self.show_start()
        return self.root

    def show_start(self):
        self.root.clear_widgets()
        self.root.add_widget(StartScreen(self.start_game))

    def start_game(self, difficulty):
        self.root.clear_widgets()
        self.root.add_widget(GameScreen(difficulty))


if __name__ == "__main__":
    PenaltyApp().run()