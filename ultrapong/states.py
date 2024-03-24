
class Game:
    alphabet = [
        "ball_bounced_on_table_1", # left side of table
        "ball_bounced_on_table_2", # right side of table
        "ball_hit_the_net",
        "ball_hit_by_player_1",
        "ball_hit_by_player_2",
        "ball_lost"
    ]

    states = [
        "p1_liable_before_hit", # player 1 liable if the ball is lost of bounces on their side
        "p1_liable_after_hit", # player 2 liable if the ball is lost of bounces on their side
        "p1_serves",
        "p1_loses",
        "p2_liable_before_hit",
        "p2_liable_after_hit",
        "p2_serves",
        "p2_loses"
    ]

    def __init__(self):
        pass

