import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer

# --- USER INPUT ---
VIDEO_PATH = '2025-12-20_04:24:44PM_flight5.avi'  # Replace with your video file path
CSV_PATH = 'second wind.csv'

# Replace with actual camera intrinsics
K = np.array([[973.36879615, 0, 642.90295147],  # fx, 0, cx
              [0, 972.01937, 327.62201959],  # 0, fy, cy
              [0,   0,   1]])

# Replace with actual extrinsics (rotation and translation from Pixhawk to camera)
R_cam_body = np.eye(3)  # Identity for now
t_cam_body = np.zeros((3, 1))

# --- Load CSV ---
df = pd.read_csv(CSV_PATH)
df = df[df['Altitude_m'] > 5].reset_index(drop=True)

# --- Coordinate Transformation (WGS84 to ENU) ---
lat0, lon0, alt0 = df.loc[0, ['Lat', 'Lon', 'Altitude_m']]
proj_pipeline = (
    f"+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
    f"+step +proj=cart +ellps=WGS84 "
    f"+step +proj=topocentric +lat_0={lat0} +lon_0={lon0} +h_0={alt0} +ellps=WGS84 +units=m"
)
transformer = Transformer.from_pipeline(proj_pipeline)

E, N, U = transformer.transform(df['Lat'].values, df['Lon'].values, df['Altitude_m'].values)

# --- Euler Angles to Rotation Matrix ---
def euler_to_rotmat(roll, pitch, yaw):
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# --- Load Video ---
video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    raise IOError("Cannot open video file")

frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
corners = np.array([[0, 0, 1], [frame_w-1, 0, 1], [frame_w-1, frame_h-1, 1], [0, frame_h-1, 1]]).T

# --- Compute Homographies ---
homographies = []
frames = []
for i, row in df.iterrows():
    frame_idx = int(row['Frame'])
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video.read()
    if not ret:
        continue
    x, y, z = E[i], N[i], U[i]
    R_body = euler_to_rotmat(row['Roll_deg'], row['Pitch_deg'], row['Yaw_deg'])
    R_cam = R_body @ R_cam_body
    t_cam = R_body @ t_cam_body + np.array([[x], [y], [z]])
    H = K @ np.hstack((R_cam[:, :2], t_cam))
    H = np.linalg.inv(H)
    homographies.append(H)
    frames.append(frame)

# --- Determine Map Bounds ---
min_x = min_y = np.inf
max_x = max_y = -np.inf
for H in homographies:
    pts = H @ corners
    pts /= pts[2]
    xs, ys = pts[0], pts[1]
    min_x = min(min_x, xs.min())
    max_x = max(max_x, xs.max())
    min_y = min(min_y, ys.min())
    max_y = max(max_y, ys.max())

canvas_width = int(np.ceil(max_x - min_x))
canvas_height = int(np.ceil(max_y - min_y))
offset = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
stitched = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# --- Warp Frames to Canvas ---
for frame, H in zip(frames, homographies):
    H_canvas = offset @ H
    warped = cv2.warpPerspective(frame, H_canvas, (canvas_width, canvas_height))
    mask = (warped > 0).any(axis=2)
    stitched[mask] = warped[mask]

cv2.imwrite('stitched_map.jpg', stitched)
print("Stitched map saved as 'stitched_map.jpg'")
