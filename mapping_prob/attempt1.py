import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm
import os
from collections import defaultdict

# --- USER INPUT ---
VIDEO_PATH = '2025-12-20_04:24:44PM_flight5.avi'
CSV_PATH = 'second wind.csv'
PIXELS_PER_METER = 5  # Adjusted automatically from altitude (~6m)
TILE_SIZE_PX = 512

# Replace with actual camera intrinsics
K = np.array([[700, 0, 640], [0, 700, 360], [0, 0, 1]])

# 90-degree rotation around Z-axis from Pixhawk heading
R_cam_body = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
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

# --- Euler to Rotation ---
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

# --- TILE STITCHING ---
tiles = defaultdict(lambda: np.zeros((TILE_SIZE_PX, TILE_SIZE_PX, 3), dtype=np.uint8))
counts = defaultdict(int)

print("Stitching into tiles...")
for i in tqdm(range(len(df))):
    row = df.loc[i]
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
    try:
        H = np.linalg.inv(H)
    except:
        continue
    if not np.isfinite(H).all():
        continue
    # Warp image to flat ground
    scale = PIXELS_PER_METER
    tile_x = int(x * scale) // TILE_SIZE_PX
    tile_y = int(y * scale) // TILE_SIZE_PX
    offset_x = tile_x * TILE_SIZE_PX
    offset_y = tile_y * TILE_SIZE_PX

    canvas = np.zeros((TILE_SIZE_PX, TILE_SIZE_PX, 3), dtype=np.uint8)
    offset = np.array([[1, 0, -offset_x / scale], [0, 1, -offset_y / scale], [0, 0, 1]])
    H_canvas = offset @ H
    try:
        warped = cv2.warpPerspective(frame, H_canvas * scale, (TILE_SIZE_PX, TILE_SIZE_PX))
    except:
        continue
    mask = (warped > 0).any(axis=2)
    tiles[(tile_x, tile_y)][mask] = warped[mask]
    counts[(tile_x, tile_y)] += 1

# --- Stitch tiles into final map ---
if not tiles:
    print("No valid tiles generated.")
    exit()
tile_keys = np.array(list(tiles.keys()))
x_min, y_min = tile_keys.min(axis=0)
x_max, y_max = tile_keys.max(axis=0)
tile_w = x_max - x_min + 1
tile_h = y_max - y_min + 1
final_map = np.zeros((tile_h * TILE_SIZE_PX, tile_w * TILE_SIZE_PX, 3), dtype=np.uint8)

for (tx, ty), tile_img in tiles.items():
    ix = tx - x_min
    iy = ty - y_min
    final_map[iy*TILE_SIZE_PX:(iy+1)*TILE_SIZE_PX,
              ix*TILE_SIZE_PX:(ix+1)*TILE_SIZE_PX] = tile_img

cv2.imwrite("stitched_map_tiled.jpg", final_map)
print("Final stitched map saved as 'stitched_map_tiled.jpg'")
