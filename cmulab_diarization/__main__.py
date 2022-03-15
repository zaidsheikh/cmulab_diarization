import argparse
import json
from .run_diarization import run_diarization as get_results

if __name__ == "__main__":
   parser = argparse.ArgumentParser("Run speaker diarization")
   parser.add_argument('--path', required=True, type=str, help='file to transcribe')
   parser.add_argument('--segments', type=str, required=True, help='speaker segments (JSON array)')
   parser.add_argument('--speakers', type=str, required=True, help='speakers corresponding to the segments')
   args = parser.parse_args()

   # audio_data/X2zqiX6yL3I.mp3
   # '[[0, 5.5], [6.5, 12], [17, 25]]'
   # '["Kyle Gass", "Sean Evans", "Jack Black"]'
   get_results(args.path, json.loads(args.segments), json.loads(args.speakers))
