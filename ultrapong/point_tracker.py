class PointTracker:
    def __init__(self):
        self.player1 = 0
        self.player2 = 0

    def update(self, result):
        if result == 1:
            self.player1 += 1
        elif result == 2:
            self.player2 += 1
            
        if self.player1 >= max(self.player2 + 2, 11):
            return 1
        elif self.player2 >= max(self.player1 + 2, 11):
            return 2
        else:
            return None
        
    def reset(self):
        self.player1 = 0
        self.player2 = 0

