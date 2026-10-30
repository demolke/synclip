#!/usr/bin/env bash
# Side-by-side comparison: retarget render | reference video, frame-synced and labelled.
#
#   side_by_side.sh RENDER.mp4 VIDEO.mp4 OUT.mp4 [VIDEO_START_FRAME=0] [HEIGHT=480] \
#                   [LABEL_RENDER="retarget"] [LABEL_VIDEO="source"]
#
# Frame sync: frame k of RENDER is paired with frame VIDEO_START_FRAME + k*VIDEO_FPS/RENDER_FPS of
# VIDEO (the video is trimmed at VIDEO_START_FRAME and resampled to the render's frame rate, so a
# preview rendered with --step N lines up too).  Both are scaled to HEIGHT; output stops at the
# shorter input.  If the render starts at scene frame 1 = mocap frame 0, VIDEO_START_FRAME is the
# video frame of mocap frame 0 (0 for the stand-in take).
set -euo pipefail
if [ $# -lt 3 ]; then sed -n '2,12p' "$0"; exit 1; fi
RENDER=$1; VIDEO=$2; OUT=$3
START=${4:-0}; H=${5:-480}; LR=${6:-retarget}; LV=${7:-source}

fps_of() { ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$1"; }
RFPS=$(fps_of "$RENDER")
VFPS=$(fps_of "$VIDEO")
FONT=$(fc-match -f '%{file}' 'DejaVu Sans:bold' 2>/dev/null || true)
FONTOPT=""
[ -n "$FONT" ] && FONTOPT="fontfile='$FONT':"
TXT="${FONTOPT}fontsize=$((H/22)):fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6:x=10:y=10"

ffmpeg -y -loglevel error -i "$RENDER" -i "$VIDEO" -filter_complex "
 [0:v]setpts=PTS-STARTPTS,scale=-2:${H},setsar=1,drawtext=${TXT}:text='${LR}  f%{frame_num}':start_number=0[a];
 [1:v]trim=start_frame=${START},setpts=PTS-STARTPTS,fps=${RFPS},scale=-2:${H},setsar=1,drawtext=${TXT}:text='${LV}  (video ${VFPS} fps from f${START})'[b];
 [a][b]hstack=inputs=2:shortest=1,format=yuv420p[v]" \
  -map "[v]" -r "${RFPS}" -c:v libx264 -crf 20 "$OUT"
echo "wrote $OUT"
