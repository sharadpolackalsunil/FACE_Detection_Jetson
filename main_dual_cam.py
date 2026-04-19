"""
main_dual_cam.py — Jetson Nano Dual-Camera Face Recognition Pipeline
=====================================================================
Architecture:
  v4l2src → nvvideoconvert (NVMM) → nvstreammux (batch=2, 640x480)
         → PGIE (YOLOv8n-face) → Tracker (IOU) → SGIE (MobileFaceNet)
         → nvmultistreamtiler → nvdsosd → nveglglessink

Key Optimizations (v2):
  1. Tracker Object-ID Cache — skip SGIE tensor math for already-identified faces
  2. Vectorized Watchlist Matrix — single matrix @ vector for instant matching
  3. Fixed multi-face tensor extraction — correct 1:1 tensor→object mapping
"""

import gi
import sys
import os
import ctypes
import math
import time
import numpy as np
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

from db_utils import load_all_embeddings, log_attendance

# ======================== GLOBALS & CONSTANTS ======================== #

COOLDOWN_SEC          = 5          # seconds between re-logging same person
SIMILARITY_THRESHOLD  = 0.40       # cosine-similarity floor
MUXER_BATCH_TIMEOUT   = 40000      # microseconds
CACHE_EVICT_INTERVAL  = 100        # frames between stale-ID cleanup
CACHE_STALE_FRAMES    = 150        # evict IDs not seen for this many frames

frame_counter  = 0                 # manual frame counter
last_logged    = {}                # {user_id: timestamp}
fps_streams    = {}                # {source_id: {last_time, frame_count, fps}}

# ---- Tracker Object-ID Cache ---- #
# {object_id: (display_name, best_score, last_frame_seen)}
track_id_cache = {}

# ======================== LOAD & BUILD WATCHLIST ======================== #

embeddings_db = load_all_embeddings()
print(f"[INFO] Loaded {len(embeddings_db)} enrolled face(s) from database")

# Pre-normalise each DB embedding once at startup
for user in embeddings_db:
    norm = np.linalg.norm(user['embedding'])
    if norm > 1e-8:
        user['embedding'] = user['embedding'] / norm
    print(f"  → {user['name']}  dim={len(user['embedding'])}  "
          f"norm={np.linalg.norm(user['embedding']):.4f}")

# ---- Build the Vectorized Watchlist Matrix ---- #
# Shape: (N, D) — one row per enrolled user, pre-normalized
# All enrolled embeddings MUST have the same dimensionality.
# The actual dimension (128 or 512) is detected automatically.
if len(embeddings_db) > 0:
    _emb_dim = len(embeddings_db[0]['embedding'])
    watchlist_matrix = np.vstack(
        [u['embedding'].reshape(1, _emb_dim) for u in embeddings_db]
    ).astype(np.float32)                            # (N, D)
    watchlist_names   = [u['name']    for u in embeddings_db]
    watchlist_ids     = [u['user_id'] for u in embeddings_db]
    print(f"[INFO] Watchlist matrix built: {watchlist_matrix.shape}  "
          f"(N={len(watchlist_names)}, D={_emb_dim})")
else:
    watchlist_matrix = None
    watchlist_names  = []
    watchlist_ids    = []
    _emb_dim         = 0
    print("[WARN] No enrolled faces — recognition disabled")


# ======================== VECTORIZED FACE MATCHING ======================== #

def match_face_vectorized(live_embedding_np):
    """
    Match a live embedding against the entire watchlist in one shot.

    Steps:
        1. L2-normalize the live query vector.
        2. Compute   scores = watchlist_matrix @ query   (dot product).
           Since both sides are unit-length, this IS cosine similarity.
        3. argmax → best match.

    Returns:
        (name, user_id, score) on success, or (None, None, best_score).
    """
    if watchlist_matrix is None or watchlist_matrix.shape[0] == 0:
        return None, None, -1.0

    norm = np.linalg.norm(live_embedding_np)
    if norm < 1e-8:
        return None, None, -1.0

    query = (live_embedding_np / norm).astype(np.float32)

    # --- Handle dimension mismatch gracefully ---
    # The SGIE model may output 512-D while enrolled embeddings are 512-D,
    # or vice-versa. We dynamically truncate/pad to align.
    q_dim = query.shape[0]
    w_dim = watchlist_matrix.shape[1]
    if q_dim != w_dim:
        min_dim = min(q_dim, w_dim)
        query_use  = query[:min_dim]
        matrix_use = watchlist_matrix[:, :min_dim]
        # Re-normalize after truncation
        qn = np.linalg.norm(query_use)
        if qn > 1e-8:
            query_use = query_use / qn
        matrix_use = matrix_use / (
            np.linalg.norm(matrix_use, axis=1, keepdims=True) + 1e-8)
    else:
        query_use  = query
        matrix_use = watchlist_matrix

    # Single vectorized dot-product: (N,D) @ (D,) → (N,)
    scores = matrix_use @ query_use
    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score > SIMILARITY_THRESHOLD:
        return watchlist_names[best_idx], watchlist_ids[best_idx], best_score
    return None, None, best_score


# ======================== PIPELINE PROBE ======================== #

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe attached to the SGIE src pad.

    For every frame → for every detected face object:
      1. Check tracker cache by object_id  →  if HIT, reuse cached name.
      2. If MISS, extract SGIE tensor, run vectorized match, populate cache.
      3. Every CACHE_EVICT_INTERVAL frames, purge stale IDs.
    """
    global frame_counter, track_id_cache

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        source_id = frame_meta.source_id
        frame_counter += 1
        now = time.time()

        # ---- FPS Tracking ---- #
        if source_id not in fps_streams:
            fps_streams[source_id] = {
                'last_time': now, 'frame_count': 0, 'fps': 0.0}
        fps_streams[source_id]['frame_count'] += 1
        elapsed = now - fps_streams[source_id]['last_time']
        if elapsed >= 1.0:
            fps_streams[source_id]['fps'] = (
                fps_streams[source_id]['frame_count'] / elapsed)
            fps_streams[source_id]['frame_count'] = 0
            fps_streams[source_id]['last_time'] = now
        current_fps = fps_streams[source_id]['fps']

        # ---- FPS Overlay ---- #
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = f"FPS: {current_fps:.1f}"
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 11
        py_nvosd_text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        debug_print = (frame_counter % 30 == 0)
        if debug_print:
            print(f"[cam {source_id}] Frame {frame_counter} | "
                  f"Faces: {frame_meta.num_obj_meta} | "
                  f"FPS: {current_fps:.1f} | "
                  f"Cache: {len(track_id_cache)} IDs")

        # ---- Track IDs currently alive in THIS frame (for eviction) ---- #
        active_ids_this_frame = set()

        # ============================================================ #
        #  OBJECT LOOP — Process each detected face                     #
        # ============================================================ #
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_id = obj_meta.object_id
            active_ids_this_frame.add(obj_id)

            # --------------------------------------------------- #
            #  CACHE HIT — Tracker ID already identified           #
            # --------------------------------------------------- #
            if obj_id in track_id_cache:
                cached_name, cached_score, _ = track_id_cache[obj_id]
                # Update last-seen frame
                track_id_cache[obj_id] = (
                    cached_name, cached_score, frame_counter)

                if cached_name is not None:
                    # ---- RECOGNISED (cached) ---- #
                    _apply_recognised_overlay(
                        obj_meta, cached_name, cached_score)

                    # Attendance logging with cooldown
                    uid = _uid_for_name(cached_name)
                    if uid is not None:
                        if (uid not in last_logged
                                or now - last_logged[uid] > COOLDOWN_SEC):
                            log_attendance(uid, source_id)
                            last_logged[uid] = now
                            print(f"[ATTENDANCE] {cached_name} "
                                  f"logged on cam {source_id}")
                else:
                    # ---- UNKNOWN (cached) ---- #
                    _apply_unknown_overlay(obj_meta)

                # Skip the expensive tensor work
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
                continue

            # --------------------------------------------------- #
            #  CACHE MISS — Must extract SGIE tensor & match       #
            # --------------------------------------------------- #
            recognised = False
            l_user_meta = obj_meta.obj_user_meta_list

            if l_user_meta is None:
                print(f"  [WARN] obj_id={obj_id}: No SGIE metadata "
                      f"(SGIE skipped! Class={obj_meta.class_id}, Box={int(obj_meta.rect_params.width)}x{int(obj_meta.rect_params.height)})")

            while l_user_meta is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                except StopIteration:
                    break

                meta_type = user_meta.base_meta.meta_type

                if meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(
                        user_meta.user_meta_data)

                    # Only accept tensors from our SGIE (gie-unique-id=2)
                    if tensor_meta.unique_id != 2:
                        try:
                            l_user_meta = l_user_meta.next
                        except StopIteration:
                            break
                        continue

                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    ptr = ctypes.cast(
                        pyds.get_ptr(layer.buffer),
                        ctypes.POINTER(ctypes.c_float))

                    # Dynamic dimension: supports both 128-D and 512-D
                    emb_dim = layer.dims.d[0]
                    if emb_dim <= 0:
                        emb_dim = layer.dims.numElements

                    # CRITICAL: np.copy() to own the data before the
                    # pointer becomes invalid on the next buffer cycle
                    live_embedding = np.copy(
                        np.ctypeslib.as_array(ptr, shape=(emb_dim,)))

                    # Normalize
                    norm = np.linalg.norm(live_embedding)
                    if norm > 1e-8:
                        live_embedding /= norm

                    print(f"  [DEBUG EMBEDDING OUT] obj_id={obj_id}  dim={emb_dim}  "
                          f"norm={np.linalg.norm(live_embedding):.4f}  "
                          f"first5={live_embedding[:5]}")

                    # Vectorized match against entire watchlist
                    name, user_id, score = match_face_vectorized(
                        live_embedding)

                    if debug_print:
                        tag = 'RECOGNISED' if name else 'UNKNOWN'
                        print(f"  [MATCH] obj_id={obj_id}  "
                              f"score={score:.4f}  thresh={SIMILARITY_THRESHOLD}  "
                              f"→ {tag}")

                    if name is not None:
                        _apply_recognised_overlay(obj_meta, name, score)
                        # Cache this tracked ID
                        track_id_cache[obj_id] = (
                            name, score, frame_counter)
                        # Attendance
                        if (user_id not in last_logged
                                or now - last_logged[user_id] > COOLDOWN_SEC):
                            log_attendance(user_id, source_id)
                            last_logged[user_id] = now
                            print(f"[ATTENDANCE] {name} "
                                  f"logged on cam {source_id}")
                        recognised = True
                    else:
                        # Cache as Unknown so we don't re-process
                        track_id_cache[obj_id] = (
                            None, score, frame_counter)

                    # We found the tensor for this obj — stop iterating
                    break

                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break

            if not recognised:
                _apply_unknown_overlay(obj_meta)
                # Ensure we cache failed/skipped SGIE inferences so they don't spam the console infinitely
                if obj_id not in track_id_cache:
                    track_id_cache[obj_id] = (None, 0.0, frame_counter)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # ---- Periodic Cache Eviction ---- #
        if frame_counter % CACHE_EVICT_INTERVAL == 0:
            stale_ids = [
                oid for oid, (_, _, last_seen) in track_id_cache.items()
                if (frame_counter - last_seen) > CACHE_STALE_FRAMES
            ]
            for oid in stale_ids:
                del track_id_cache[oid]
            if stale_ids and debug_print:
                print(f"  [CACHE] Evicted {len(stale_ids)} stale ID(s), "
                      f"remaining: {len(track_id_cache)}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ======================== OVERLAY HELPERS ======================== #

def _apply_recognised_overlay(obj_meta, name, score):
    """Green box + name label for recognised faces."""
    obj_meta.text_params.display_text = f"{name} ({score:.2f})"
    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
    obj_meta.rect_params.border_width = 4
    obj_meta.text_params.font_params.font_name  = "Serif"
    obj_meta.text_params.font_params.font_size   = 12
    obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.0, 0.4, 0.0, 0.6)


def _apply_unknown_overlay(obj_meta):
    """Red box + 'Unknown' label for unrecognised faces."""
    obj_meta.text_params.display_text = "Unknown"
    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
    obj_meta.rect_params.border_width = 3
    obj_meta.text_params.font_params.font_name  = "Serif"
    obj_meta.text_params.font_params.font_size   = 12
    obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.6, 0.0, 0.0, 0.6)


def _uid_for_name(name):
    """Reverse-lookup user_id by name from the watchlist."""
    for i, n in enumerate(watchlist_names):
        if n == name:
            return watchlist_ids[i]
    return None


# ======================== SOURCE BIN HELPERS ======================== #

def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                print("Failed to link decoder src pad to source bin ghost pad")
        else:
            print("Error: Decodebin did not pick nvidia decoder plugin.")


def create_source_bin(index, uri):
    bin_name = f"source-bin-{index}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin\n")

    if uri.startswith("rtsp://"):
        uri_decode_bin = Gst.ElementFactory.make(
            "uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write("Unable to create uri decode bin\n")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect(
            "child-added",
            lambda child_proxy, obj, name, user_data:
                obj.set_property("drop-on-latency", True)
                if name.find("decodebin") != -1 else None,
            nbin)
        Gst.Bin.add(nbin, uri_decode_bin)
    else:
        # V4L2 USB Camera (e.g. /dev/video0)
        v4l2src = Gst.ElementFactory.make("v4l2src", f"v4l2src_{index}")
        v4l2src.set_property("device", uri)

        # Hardware-accelerated converter
        nvvidconv = Gst.ElementFactory.make(
            "nvvideoconvert", f"nvvidconv_{index}")
        capsfilter = Gst.ElementFactory.make("capsfilter", f"caps_{index}")

        # 640x480 to keep Nano GPU usage reasonable
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12")
        capsfilter.set_property("caps", caps)

        Gst.Bin.add(nbin, v4l2src)
        Gst.Bin.add(nbin, nvvidconv)
        Gst.Bin.add(nbin, capsfilter)

        v4l2src.link(nvvidconv)
        nvvidconv.link(capsfilter)

    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))

    if not uri.startswith("rtsp://"):
        nbin.get_static_pad("src").set_target(
            capsfilter.get_static_pad("src"))

    return nbin


# ======================== MAIN ======================== #

def main(args):
    if len(args) < 2:
        print("Usage: python3 main_dual_cam.py <cam_uri> [cam2_uri ...]")
        print("  Example: python3 main_dual_cam.py /dev/video0")
        sys.exit(1)

    sources = args[1:]
    num_sources = len(sources)

    Gst.init(None)

    print("=" * 60)
    print("  JETSON NANO — Face Recognition Pipeline  (v2)")
    print("=" * 60)
    print(f"  Sources         : {num_sources}")
    print(f"  Enrolled Faces  : {len(embeddings_db)}")
    print(f"  Threshold       : {SIMILARITY_THRESHOLD}")
    print(f"  Watchlist Matrix: "
          f"{'READY ' + str(watchlist_matrix.shape) if watchlist_matrix is not None else 'EMPTY'}")
    print(f"  Cache Eviction  : every {CACHE_EVICT_INTERVAL} frames "
          f"(stale > {CACHE_STALE_FRAMES} frames)")
    print("=" * 60)

    pipeline = Gst.Pipeline()

    # ---- Stream Muxer ---- #
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 640)
    streammux.set_property('height', 480)
    streammux.set_property('batch-size', num_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT)
    streammux.set_property('nvbuf-memory-type', 0)
    pipeline.add(streammux)

    # ---- Source Bins ---- #
    for i in range(num_sources):
        source_bin = create_source_bin(i, sources[i])
        pipeline.add(source_bin)
        padname = f"sink_{i}"
        sinkpad = streammux.get_request_pad(padname)
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    # ---- Primary GIE (YOLOv8-face) ---- #
    print("Creating Primary GIE (YOLOv8 Face)...")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "configs/pgie_config.txt")

    # ---- Tracker (IOU) ---- #
    print("Creating Tracker...")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property(
        'll-lib-file',
        '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', 'configs/tracker_config.yml')
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 640)

    # ---- Secondary GIE (MobileFaceNet) ---- #
    print("Creating Secondary GIE (MobileFaceNet)...")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine")
    sgie.set_property('config-file-path', "configs/sgie_config.txt")

    # ---- Tiler ---- #
    print("Creating Tiler...")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler_rows = int(math.sqrt(num_sources))
    tiler_columns = int(math.ceil(float(num_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", 1280)
    tiler.set_property("height", 720)

    # ---- Converter, OSD, Sink ---- #
    print("Creating Converter & OSD & Sink...")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property('sync', False)
    sink.set_property('qos', False)

    # ---- Add everything to pipeline ---- #
    print("Adding elements to Pipeline...")
    for elem in [pgie, tracker, sgie, tiler, nvvidconv, nvosd]:
        pipeline.add(elem)
    if transform:
        pipeline.add(transform)
    pipeline.add(sink)

    # ---- Link the pipeline ---- #
    print("Linking elements in Pipeline...")
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie)
    sgie.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)

    if transform:
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # ---- Probe on SGIE source pad ---- #
    sgie_src_pad = sgie.get_static_pad("src")
    if not sgie_src_pad:
        sys.stderr.write("Unable to get src pad of sgie\n")
    else:
        sgie_src_pad.add_probe(
            Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # ---- Run ---- #
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n[EOS] End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write(f"[WARNING] {err}: {debug}\n")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write(f"[ERROR] {err}: {debug}\n")
            loop.quit()
        return True

    bus.connect("message", bus_call, loop)

    print("\nStarting pipeline — press Ctrl+C to stop\n")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except BaseException:
        pass

    print("Exiting app...")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))