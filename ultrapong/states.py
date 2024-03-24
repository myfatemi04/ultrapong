
class MatchState:
    states = [
        "start",
        "p1_serves",
        "p1_liable_before_hit", # player 1 liable if the ball is lost of bounces on their side
        "p1_liable_after_hit", # player 2 liable if the ball is lost of bounces on their side
        "p1_loses",
        "p2_serves",
        "p2_liable_before_hit",
        "p2_liable_after_hit",
        "p2_loses",
        "reset" # game needs to be reset
    ]

    start_state = "start"

    final_states = {"p1_loses", "p2_loses", "reset"}

    alphabet = [
        "ball_bounced_on_table_1", # left side of table
        "ball_bounced_on_table_2", # right side of table
        "ball_hit_by_player_1",
        "ball_hit_by_player_2",
        "ball_lost"
    ]

    transitions = { # (start state, letter) -> new state
        ("start", "ball_bounced_on_table_1") : "p1_liable_after_hit",
        ("start", "ball_bounced_on_table_2") : "p2_liable_after_hit", # some leniency in detection time
        ("start", "ball_hit_by_player_1") : "p1_serves",
        ("start", "ball_hit_by_player_2") : "p2_serves",
        ("start", "ball_lost") : "reset", # invalid

        ("p1_serves", "ball_bounced_on_table_1") : "p1_liable_after_hit",
        ("p1_serves", "ball_bounced_on_table_2") : "p1_loses",
        ("p1_serves", "ball_hit_by_player_1") : "p1_loses",
        ("p1_serves", "ball_hit_by_player_2") : "p2_loses",
        ("p1_serves", "ball_lost") : "p1_loses",

        ("p2_serves", "ball_bounced_on_table_1") : "p2_loses",
        ("p2_serves", "ball_bounced_on_table_2") : "p2_liable_after_hit",
        ("p2_serves", "ball_hit_by_player_1") : "p1_loses",
        ("p2_serves", "ball_hit_by_player_2") : "p2_loses",
        ("p2_serves", "ball_lost") : "p2_loses",

        ("p1_liable_after_hit", "ball_bounced_on_table_1") : "p1_loses",
        ("p1_liable_after_hit", "ball_bounced_on_table_2") : "p2_liable_before_hit",
        ("p1_liable_after_hit", "ball_hit_by_player_1") : "p1_loses",
        ("p1_liable_after_hit", "ball_hit_by_player_2") : "p2_loses",
        ("p1_liable_after_hit", "ball_lost") : "p1_loses",

        ("p2_liable_after_hit", "ball_bounced_on_table_1") : "p1_liable_before_hit",
        ("p2_liable_after_hit", "ball_bounced_on_table_2") : "p2_loses",
        ("p2_liable_after_hit", "ball_hit_by_player_1") : "p1_loses",
        ("p2_liable_after_hit", "ball_hit_by_player_2") : "p2_loses",
        ("p2_liable_after_hit", "ball_lost") : "p2_loses",

        ("p1_liable_before_hit", "ball_bounced_on_table_1") : "p1_loses",
        ("p1_liable_before_hit", "ball_bounced_on_table_2") : "p1_loses",
        ("p1_liable_before_hit", "ball_hit_by_player_1") : "p1_liable_after_hit",
        ("p1_liable_before_hit", "ball_hit_by_player_2") : "p2_loses",
        ("p1_liable_before_hit", "ball_lost") : "p1_loses",

        ("p2_liable_before_hit", "ball_bounced_on_table_1") : "p2_loses",
        ("p2_liable_before_hit", "ball_bounced_on_table_2") : "p2_loses",
        ("p2_liable_before_hit", "ball_hit_by_player_1") : "p1_loses",
        ("p2_liable_before_hit", "ball_hit_by_player_2") : "p2_liable_after_hit",
        ("p2_liable_before_hit", "ball_lost") : "p2_loses",
    }

    def __init__(self, initial_state="start"):
        self._current_state = initial_state

    def current_state(self):
        return self._current_state

    # Transitions current state and returns the new one
    def transition(self, letter):
        key = (self._current_state, letter)
        if key not in self.transitions.keys():
            input("Current state:", self._current_state, "...")
        self._current_state = self.transitions[key]
        return self._current_state
