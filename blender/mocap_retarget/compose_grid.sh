#!/usr/bin/env bash
# 2x2 review video, all panels frame-synced (render frame k = video frame START+k):
#   top-left: rig from the calibrated camera   | top-right: that camera's video
#   bottom-left: rig composited over the video | bottom-right: tracking 3/4 view of the rig
#
#   compose_grid.sh CALIB.mp4 OVERLAY.mov TRACK.mp4 VIDEO.mp4 OUT.mp4 [CAM=camB] [START=0] [H=480] [CRF=23]
# CALIB / OVERLAY / TRACK come from render_preview.py (--calib rig.json:CAM, --calib ... --overlay
# --no-stamp, --view track34).
set -euo pipefail
if [ $# -lt 5 ]; then sed -n '2,9p' "$0"; exit 1; fi
CAL=$1; OVL=$2; TRK=$3; VID=$4; OUT=$5; CAM=${6:-camB}; START=${7:-0}; H=${8:-480}; CRF=${9:-23}
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$CAL")
FONT=$(fc-match -f '%{file}' 'DejaVu Sans:bold' 2>/dev/null || true)
T="fontfile='${FONT}':fontsize=$((H/22)):fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6:x=10:y=10"
ffmpeg -y -loglevel error -i "$CAL" -i "$OVL" -i "$TRK" -i "$VID" -filter_complex "
 [3:v]trim=start_frame=${START},setpts=PTS-STARTPTS,fps=${FPS},scale=-2:${H},setsar=1,split=2[v1][v2];
 [0:v]setpts=PTS-STARTPTS,scale=-2:${H},setsar=1,drawtext=${T}:text='retarget - ${CAM} calibrated camera'[a];
 [v1]drawtext=${T}:text='${CAM} video  f%{frame_num}':start_number=${START}[b];
 [1:v]setpts=PTS-STARTPTS,scale=-2:${H},format=rgba,colorchannelmixer=aa=0.75[o];
 [v2][o]overlay=shortest=1,drawtext=${T}:text='rig over ${CAM} video'[c];
 [2:v]setpts=PTS-STARTPTS,scale=-2:${H},setsar=1,drawtext=${T}:text='rig - tracking 3/4 view'[d];
 [a][b]hstack=inputs=2:shortest=1[top];[c][d]hstack=inputs=2:shortest=1[bot];
 [top][bot]vstack=inputs=2:shortest=1,format=yuv420p[v]" \
 -map "[v]" -r "$FPS" -c:v libx264 -preset medium -crf "$CRF" -movflags +faststart "$OUT"
echo "wrote $OUT"
