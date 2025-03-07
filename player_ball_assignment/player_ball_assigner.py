import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70 

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        for player_id , player in players.items():
            player_bbox = player['bbox']

            minimum_distance = 99999
            assigned_player = -1 # player with ball possession

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position) # left foot (bottom left corner in bbox)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position) # right foot (bottom right corner in bbox)
            distance = min(distance_left, distance_right)


            if distance < self.max_player_ball_distance: # player within acceptable range
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id # closest eligible player

        return assigned_player



