import gi
import sys
import ctypes
import math
import time
import numpy as np
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

from db_utils import load_all_embeddings, log_attendance

# -------- Try CuPy for GPU-accelerated matching, fallback to NumPy -------- #
try:
    import cupy as cp
    GPU_MATCHING = True
    print("[INFO] CuPy found — embedding matching will run on GPU")
except ImportError:
    GPU_MATCHING = False
    print("[WARN] CuPy not found — embedding matching will run on CPU (NumPy)")

# ----------------- GLOBALS & CONSTANTS ----------------- #
COOLDOWN_SEC = 300        # 5 minutes between re-logging same person
SIMILARITY_THRESHOLD = 0.40
MUXER_BATCH_TIMEOUT_USEC = 40000

last_logged = {}  # Format: {user_id: timestamp_in_seconds}

# Load embeddings from SQLite and optionally move them to GPU
_raw_db = load_all_embeddings()

if GPU_MATCHING and len(_raw_db) > 0:
    # Build a single GPU matrix (N x dim) for batched cosine similarity
    _emb_matrix_np = np.stack([u['embedding'] for u in _raw_db], axis=0)  # (N, 128/512)
    emb_matrix_gpu = cp.asarray(_emb_matrix_np, dtype=cp.float32)         # on GPU
    # Pre-compute norms once
    emb_norms_gpu = cp.linalg.norm(emb_matrix_gpu, axis=1, keepdims=True)
    # Avoid division by zero
    emb_norms_gpu = cp.maximum(emb_norms_gpu, 1e-8)
    emb_matrix_gpu_normed = emb_matrix_gpu / emb_norms_gpu
    embeddings_db = _raw_db  # keep metadata (names, ids)
    print(f"[INFO] Loaded {len(embeddings_db)} face embeddings onto GPU")
else:
    emb_matrix_gpu_normed = None
    embeddings_db = _raw_db
    print(f"[INFO] Loaded {len(embeddings_db)} face embeddings (CPU)")


# ----------------- GPU-ACCELERATED MATCHING ----------------- #

def match_face_gpu(live_embedding_np):
    """
    Matches a live embedding against the database using CuPy (GPU).
    Returns (name, user_id, score) or (None, None, -1).
    """
    if len(embeddings_db) == 0:
        return None, None, -1.0

    # Transfer single embedding to GPU and normalise
    live_gpu = cp.asarray(live_embedding_np, dtype=cp.float32).reshape(1, -1)
    live_norm = cp.linalg.norm(live_gpu)
    if live_norm < 1e-8:
        return None, None, -1.0
    live_gpu /= live_norm

    # Batched cosine similarity: (1, dim) @ (dim, N) -> (1, N)
    scores = cp.dot(live_gpu, emb_matrix_gpu_normed.T).flatten()  # (N,)

    best_idx = int(cp.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score > SIMILARITY_THRESHOLD:
        user = embeddings_db[best_idx]
        return user['name'], user['user_id'], best_score
    return None, None, best_score


def match_face_cpu(live_embedding_np):
    """
    Fallback: CPU-based matching with NumPy.
    Returns (name, user_id, score) or (None, None, -1).
    """
    if len(embeddings_db) == 0:
        return None, None, -1.0

    norm = np.linalg.norm(live_embedding_np)
    if norm < 1e-8:
        return None, None, -1.0
    live = live_embedding_np / norm

    best_score = -1.0
    best_match = None
    for user in embeddings_db:
        score = float(np.dot(live, user['embedding']) /
                       (np.linalg.norm(user['embedding']) + 1e-8))
        if score > best_score:
            best_score = score
            best_match = user

    if best_match and best_score > SIMILARITY_THRESHOLD:
        return best_match['name'], best_match['user_id'], best_score
    return None, None, best_score


# Pick the right implementation
match_face = match_face_gpu if GPU_MATCHING else match_face_cpu


# ----------------- PIPELINE PROBE ----------------- #

def osd_sink_pad_buffer_probe(pad, info, u_data):
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

        # Periodic status log
        if frame_meta.frame_num % 30 == 0:
            print(f"[cam {frame_meta.source_id}] Frame {frame_meta.frame_num} | "
                  f"Faces: {frame_meta.num_obj_meta}")

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # ---- Attempt Recognition via SGIE tensor output ---- #
            recognised = False
            l_user_meta = obj_meta.obj_user_meta_list
            while l_user_meta is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                    if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type(
                            "NVDSINFER_TENSOR_OUTPUT_META"):
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(
                            user_meta.user_meta_data)
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                        ptr = ctypes.cast(
                            pyds.get_ptr(layer.buffer),
                            ctypes.POINTER(ctypes.c_float))
                        v = np.ctypeslib.as_array(
                            ptr, shape=(layer.inferDims.d[0],))
                        live_embedding = np.copy(v)

                        name, user_id, score = match_face(live_embedding)

                        if name is not None:
                            # ---- RECOGNISED ---- #
                            obj_meta.text_params.display_text = (
                                f"{name} ({score:.2f})")
                            obj_meta.rect_params.border_color.set(
                                0.0, 1.0, 0.0, 1.0)  # Green
                            obj_meta.rect_params.border_width = 4

                            # Text styling
                            obj_meta.text_params.font_params.font_name = "Serif"
                            obj_meta.text_params.font_params.font_size = 12
                            obj_meta.text_params.font_params.font_color.set(
                                1.0, 1.0, 1.0, 1.0)  # White text
                            obj_meta.text_params.set_bg_clr = 1
                            obj_meta.text_params.text_bg_clr.set(
                                0.0, 0.4, 0.0, 0.6)  # Dark green bg

                            # Attendance logging with cooldown
                            now = time.time()
                            if (user_id not in last_logged or
                                    now - last_logged[user_id] > COOLDOWN_SEC):
                                log_attendance(
                                    user_id, frame_meta.source_id)
                                last_logged[user_id] = now
                                print(f"[ATTENDANCE] {name} logged on "
                                      f"cam {frame_meta.source_id}")

                            recognised = True
                        # If name is None, we fall through to "Unknown"
                except Exception as e:
                    print(f"[PROBE ERROR] {e}")

                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break

            # ---- NOT RECOGNISED  →  Show "Unknown" ---- #
            if not recognised:
                obj_meta.text_params.display_text = "Unknown"
                obj_meta.rect_params.border_color.set(
                    1.0, 0.0, 0.0, 1.0)  # Red
                obj_meta.rect_params.border_width = 3

                obj_meta.text_params.font_params.font_name = "Serif"
                obj_meta.text_params.font_params.font_size = 12
                obj_meta.text_params.font_params.font_color.set(
                    1.0, 1.0, 1.0, 1.0)  # White text
                obj_meta.text_params.set_bg_clr = 1
                obj_meta.text_params.text_bg_clr.set(
                    0.6, 0.0, 0.0, 0.6)  # Dark red bg

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ----------------- SOURCE BIN HELPERS ----------------- #

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
        vidconv = Gst.ElementFactory.make("videoconvert", f"vidconv_{index}")
        nvvidconv = Gst.ElementFactory.make(
            "nvvideoconvert", f"nvvidconv_{index}")
        capsfilter = Gst.ElementFactory.make("capsfilter", f"caps_{index}")
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=(string)NV12")
        capsfilter.set_property("caps", caps)

        Gst.Bin.add(nbin, v4l2src)
        Gst.Bin.add(nbin, vidconv)
        Gst.Bin.add(nbin, nvvidconv)
        Gst.Bin.add(nbin, capsfilter)

        v4l2src.link(vidconv)
        vidconv.link(nvvidconv)
        nvvidconv.link(capsfilter)

    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))

    if not uri.startswith("rtsp://"):
        nbin.get_static_pad("src").set_target(
            capsfilter.get_static_pad("src"))

    return nbin


# ----------------- MAIN EXECUTION ----------------- #

def main(args):
    if len(args) < 2:
        print("Usage: python3 main_dual_cam.py <cam1_uri> [cam2_uri ...]")
        print("  Examples:")
        print("    python3 main_dual_cam.py /dev/video0")
        print("    python3 main_dual_cam.py /dev/video0 /dev/video1")
        sys.exit(1)

    sources = args[1:]
    num_sources = len(sources)

    Gst.init(None)

    print("=" * 60)
    print("  JETSON NANO — GPU Face Recognition Pipeline")
    print("=" * 60)
    print(f"  Sources       : {num_sources}")
    print(f"  GPU Matching  : {'YES (CuPy)' if GPU_MATCHING else 'NO (NumPy CPU)'}")
    print(f"  Enrolled Faces: {len(embeddings_db)}")
    print(f"  Threshold     : {SIMILARITY_THRESHOLD}")
    print("=" * 60)

    pipeline = Gst.Pipeline()

    # ---- Stream Muxer ---- #
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', num_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    # Keep buffers in GPU memory
    streammux.set_property('nvbuf-memory-type', 0)  # 0 = NVBUF_MEM_DEFAULT (GPU)
    pipeline.add(streammux)

    # ---- Source Bins ---- #
    for i in range(num_sources):
        source_bin = create_source_bin(i, sources[i])
        if not source_bin:
            sys.stderr.write("Unable to create source bin\n")
        pipeline.add(source_bin)
        padname = f"sink_{i}"
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin\n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin\n")
        srcpad.link(sinkpad)

    # ---- Primary GIE (YOLOv8-face) — runs on GPU via TensorRT ---- #
    print("Creating Primary GIE (YOLOv8 Face)...")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "configs/pgie_config.txt")

    # ---- Tracker (IOU tracker — lightweight, GPU-friendly) ---- #
    print("Creating Tracker...")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property(
        'll-lib-file',
        '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', 'configs/tracker_config.yml')
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 640)

    # ---- Secondary GIE (MobileFaceNet) — runs on GPU via TensorRT ---- #
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

    # EGL sink for Jetson display
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

    # ---- Link the full GPU pipeline ---- #
    # streammux → pgie → tracker → sgie → tiler → nvvidconv → nvosd → sink
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

    # ---- Probe on OSD for recognition logic ---- #
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write("Unable to get sink pad of nvosd\n")
    else:
        osd_sink_pad.add_probe(
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
