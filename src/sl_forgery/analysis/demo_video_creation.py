from pathlib import Path
from visualize_rois import generate_aligned_video


# Choose session
project = "MaalstroomicFlow"
mouse_id = "15"
session = "2025-07-23-12-19-00-225677"


# Path to data
workdir_path = Path("/local/workdir")
session_path = workdir_path / "sun_data" / project / mouse_id / session

# Change this to where you want to save the file
out_path = Path("/local/workdir/jrg349/sl-forgery/src/sl_forgery/analysis") / "meso_behavior_aligned.mp4"

# This function displays torque data on the plot, and can be easily adapted to display any data as long as it has time stamps
# This function takes around 1 hour to run
generate_aligned_video(
    session_path, 
    out_path=out_path, 
    window_pre=int(3e6), window_post=int(.5e6), 
    camera="left"
)