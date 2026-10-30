"""
Small, dependency-free H.264 (Annex-B) helpers for recordings.

* :func:`nal_units` / :func:`nal_types` -- fast start-code scan (``bytes.find``,
  so it is cheap even for large access units, unlike a per-byte Python loop).
* :func:`codec_config` -- the SPS/PPS NAL units of an access unit (what a
  decoder needs before the first IDR).
* :func:`parse_sps` -- coded picture size (with frame cropping applied) and,
  when the encoder wrote VUI timing info, the nominal frame rate.

Pure functions, no PyAV needed.
"""

from __future__ import annotations

NAL_SLICE = 1
NAL_IDR = 5
NAL_SEI = 6
NAL_SPS = 7
NAL_PPS = 8
NAL_AUD = 9

_HIGH_PROFILES = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}


def nal_units(data: bytes) -> list[tuple[int, int, int]]:
    """``(type, start, end)`` of every NAL unit in an Annex-B buffer.

    *start* is the offset of the start code (``00 00 01`` or ``00 00 00 01``),
    *end* the offset of the next start code (or ``len(data)``), so
    ``data[start:end]`` is the NAL unit **including** its start code.
    """
    out: list[tuple[int, int, int]] = []
    n = len(data)
    i = data.find(b"\x00\x00\x01")
    starts: list[tuple[int, int]] = []        # (start-code offset, header offset)
    while i >= 0 and i + 3 < n + 1:
        sc = i - 1 if i > 0 and data[i - 1] == 0 else i
        starts.append((sc, i + 3))
        i = data.find(b"\x00\x00\x01", i + 3)
    for k, (sc, hdr) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else n
        if hdr < n:
            out.append((data[hdr] & 0x1F, sc, end))
    return out


def nal_types(data: bytes) -> list[int]:
    """NAL unit types present in an Annex-B access unit."""
    return [t for t, _, _ in nal_units(data)]


def has_slice(data: bytes) -> bool:
    """True if the access unit carries a picture (IDR or non-IDR slice)."""
    return any(t in (NAL_SLICE, NAL_IDR) for t in nal_types(data))


def is_idr(data: bytes) -> bool:
    return NAL_IDR in nal_types(data)


def codec_config(data: bytes) -> bytes:
    """The SPS + PPS NAL units of *data* (Annex-B, with 4-byte start codes)."""
    parts = []
    for t, s, e in nal_units(data):
        if t in (NAL_SPS, NAL_PPS):
            nal = data[s:e].lstrip(b"\x00")          # drop the start code's zeros
            parts.append(b"\x00\x00\x00" + nal)      # ... and write a 4-byte one
    return b"".join(parts)


# ---------------------------------------------------------------- SPS parse
def _rbsp(nal_payload: bytes) -> bytes:
    """Remove emulation-prevention bytes (``00 00 03`` -> ``00 00``)."""
    out = bytearray()
    zeros = 0
    for b in nal_payload:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


class _Bits:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def u(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]           # IndexError past the end
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("bad Exp-Golomb code")
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def _skip_scaling_list(b: _Bits, size: int) -> None:
    last = nxt = 8
    for _ in range(size):
        if nxt != 0:
            nxt = (last + b.se() + 256) % 256
        last = nxt if nxt != 0 else last


def parse_sps(data: bytes) -> dict | None:
    """Parse the first SPS in an Annex-B buffer.

    Returns ``{"width", "height", "profile_idc", "level_idc", "fps"}`` (``fps``
    from VUI timing info, ``None`` when absent) or ``None`` if *data* holds no
    (parsable) SPS.
    """
    for t, s, e in nal_units(data):
        if t != NAL_SPS:
            continue
        nal = data[s:e].lstrip(b"\x00")[1:]           # start code '01' -> header
        try:
            return _parse_sps_rbsp(_rbsp(nal[1:]))     # skip the NAL header byte
        except (IndexError, ValueError):
            return None
    return None


def _parse_sps_rbsp(r: bytes) -> dict:
    b = _Bits(r)
    profile = b.u(8)
    b.u(8)                                            # constraint flags
    level = b.u(8)
    b.ue()                                            # seq_parameter_set_id
    chroma_format_idc = 1
    separate_colour_plane = 0
    if profile in _HIGH_PROFILES:
        chroma_format_idc = b.ue()
        if chroma_format_idc == 3:
            separate_colour_plane = b.u(1)
        b.ue()                                        # bit_depth_luma_minus8
        b.ue()                                        # bit_depth_chroma_minus8
        b.u(1)                                        # qpprime_y_zero_transform_bypass
        if b.u(1):                                    # seq_scaling_matrix_present
            for i in range(8 if chroma_format_idc != 3 else 12):
                if b.u(1):
                    _skip_scaling_list(b, 16 if i < 6 else 64)
    b.ue()                                            # log2_max_frame_num_minus4
    poc_type = b.ue()
    if poc_type == 0:
        b.ue()
    elif poc_type == 1:
        b.u(1)
        b.se()
        b.se()
        for _ in range(b.ue()):
            b.se()
    b.ue()                                            # max_num_ref_frames
    b.u(1)                                            # gaps_in_frame_num_allowed
    w_mbs = b.ue() + 1
    h_map = b.ue() + 1
    frame_mbs_only = b.u(1)
    if not frame_mbs_only:
        b.u(1)                                        # mb_adaptive_frame_field
    b.u(1)                                            # direct_8x8_inference
    crop = (0, 0, 0, 0)
    if b.u(1):
        crop = (b.ue(), b.ue(), b.ue(), b.ue())       # left, right, top, bottom
    chroma_array_type = 0 if separate_colour_plane else chroma_format_idc
    if chroma_array_type == 0:
        cux, cuy = 1, 2 - frame_mbs_only
    else:
        sub_w = 1 if chroma_format_idc == 3 else 2
        sub_h = 2 if chroma_format_idc == 1 else 1
        cux, cuy = sub_w, sub_h * (2 - frame_mbs_only)
    width = w_mbs * 16 - cux * (crop[0] + crop[1])
    height = (2 - frame_mbs_only) * h_map * 16 - cuy * (crop[2] + crop[3])
    fps = None
    try:
        if b.u(1):                                    # vui_parameters_present
            if b.u(1):                                # aspect_ratio_info_present
                if b.u(8) == 255:
                    b.u(16)
                    b.u(16)
            if b.u(1):                                # overscan_info_present
                b.u(1)
            if b.u(1):                                # video_signal_type_present
                b.u(3)
                b.u(1)
                if b.u(1):
                    b.u(24)
            if b.u(1):                                # chroma_loc_info_present
                b.ue()
                b.ue()
            if b.u(1):                                # timing_info_present
                units, scale = b.u(32), b.u(32)
                if units:
                    fps = scale / (2.0 * units)
    except (IndexError, ValueError):
        fps = None
    return {"width": int(width), "height": int(height), "profile_idc": profile,
            "level_idc": level, "fps": fps}
