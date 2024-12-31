from utils import read_video, save_video
from tracking import Tracker
import cv2 
from team_assigner import TeamAssigner

def main():
    # 1- read the input video
    input_path = 'input_vids/08fd33_4.mp4'
    video_frames = read_video(input_path)
    print(f"Loaded {len(video_frames)} frames from {input_path}")

    # 2- initialize Tracker class
    tracker = Tracker('models/best.pt')
    print("Tracker initialized with YOLO model.")

    # 3- generate or load object tracks
    stub_path = 'stubs/track_stubs.pkl'
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True, 
                                       stub_path=stub_path)
    print("Object tracking completed.")
    
    # 4- assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num , player_track in enumerate(tracks['players']):
        for player_id , track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track["bbox"],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    print("Teams assigned successfully!")
 
    # 5 - draw annotations on video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    print("Annotations drawn on video frames.")

    # 6- save the annotated video
    output_path = 'output_videos/better_annoTracks_withColor.avi'
    save_video(output_video_frames, output_path)
    print(f"Annotated video saved to {output_path}")

if __name__ == '__main__':
    main()