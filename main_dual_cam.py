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

# ----------------- GLOBALS & CONSTANTS ----------------- #
embeddings_db = load_all_embeddings()
last_logged = {}  # Format: {user_id: timestamp_in_seconds}
COOLDOWN_SEC = 300  # 5 minutes
SIMILARITY_THRESHOLD = 0.60
MUXER_BATCH_TIMEOUT_USEC = 40000

# ----------------- PIPELINE UTILS ----------------- #

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe attached to OSD sink pad (or SGIE src pad).
    Extracts the user metadata containing tensor outputs from the SGIE,
    performs cosine similarity against the database, updates the OSD text,
    and logs attendance if recognized.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
            
        camera_id = frame_meta.pad_index
            
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
                
            # Iterate through all user metadata attached to the object
            l_user_meta = obj_meta.obj_user_meta_list
            recognized = False
            
            while l_user_meta is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                except StopIteration:
                    break
                    
                # We are looking for NVDSINFER_TENSOR_OUTPUT_META attached by SGIE
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVDSINFER_TENSOR_OUTPUT_META"):
                    try:
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        
                        # Assuming the SGIE produces 1 output layer (the embedding)
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                        
                        # Fetching the pointer to the inference output
                        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                        
                        # Typically 128D or 512D
                        embed_dim = layer.inferDims.d[0]
                        v = np.ctypeslib.as_array(ptr, shape=(embed_dim,))
                        
                        # Copy and normalize the embedding
                        live_embedding = np.copy(v)
                        norm = np.linalg.norm(live_embedding)
                        if norm > 0:
                            live_embedding = live_embedding / norm
                            
                            best_match = None
                            best_score = -1
                            
                            for user in embeddings_db:
                                score = cosine_similarity(live_embedding, user['embedding'])
                                if score > best_score:
                                    best_score = score
                                    best_match = user
                                    
                            if best_match and best_score > SIMILARITY_THRESHOLD:
                                # Recognized!
                                name = best_match['name']
                                user_id = best_match['user_id']
                                
                                # Update Display Text (OSD)
                                obj_meta.text_params.display_text = f"{name} ({best_score:.2f})"
                                
                                # Set bounding box color to green for recognized
                                obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                                
                                recognized = True
                                
                                # Attendance Logging with Cooldown
                                now = time.time()
                                user_cam_key = f"{user_id}_{camera_id}"
                                
                                if user_cam_key not in last_logged or (now - last_logged[user_cam_key]) > COOLDOWN_SEC:
                                    log_attendance(user_id, camera_id)
                                    last_logged[user_cam_key] = now
                                    print(f"--> Logged attendance: {name} on Camera {camera_id}")

                    except Exception as e:
                        print("Error parsing tensor:", e)
                        
                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break
                    
            if not recognized:
                # Set bounding box color to red for unrecognized
                obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                obj_meta.text_params.display_text = "Unknown"

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not audio.
    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                print("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            print("Error: Decodebin did not pick nvidia decoder plugin.")

def create_source_bin(index, uri):
    bin_name = f"source-bin-{index}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin \n")

    if uri.startswith("rtsp://"):
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write("Unable to create uri decode bin \n")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", lambda child_proxy, obj, name, user_data: 
            obj.set_property("drop-on-latency", True) if name.find("decodebin") != -1 else None, nbin)
        Gst.Bin.add(nbin, uri_decode_bin)
    else:
        # Assume V4L2 USB Camera (e.g. /dev/video0)
        v4l2src = Gst.ElementFactory.make("v4l2src", f"v4l2src_{index}")
        v4l2src.set_property("device", uri)
        vidconv = Gst.ElementFactory.make("videoconvert", f"vidconv_{index}")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv_{index}")
        capsfilter = Gst.ElementFactory.make("capsfilter", f"caps_{index}")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)NV12")
        capsfilter.set_property("caps", caps)
        
        Gst.Bin.add(nbin, v4l2src)
        Gst.Bin.add(nbin, vidconv)
        Gst.Bin.add(nbin, nvvidconv)
        Gst.Bin.add(nbin, capsfilter)
        
        v4l2src.link(vidconv)
        vidconv.link(nvvidconv)
        nvvidconv.link(capsfilter)
        
        bin_pad = capsfilter.get_static_pad("src")
        
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    
    if not uri.startswith("rtsp://"):
        nbin.get_static_pad("src").set_target(capsfilter.get_static_pad("src"))

    return nbin

# ----------------- MAIN EXECUTION ----------------- #

def main(args):
    # Pass RTSP URLs or V4L2 device nodes as args
    if len(args) < 2:
        print("Usage: python3 main_dual_cam.py <cam1_uri> <cam2_uri>")
        print("Example: python3 main_dual_cam.py /dev/video0 /dev/video1")
        sys.exit(1)

    sources = args[1:]
    num_sources = len(sources)

    Gst.init(None)

    print("Creating Pipeline...")
    pipeline = Gst.Pipeline()

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', num_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    pipeline.add(streammux)

    for i in range(num_sources):
        source_bin = create_source_bin(i, sources[i])
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = f"sink_{i}"
        sinkpad = streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    print("Creating Primary GIE (YOLOv8 Face)...")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "configs/pgie_config.txt")

    print("Creating Tracker...")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', 'configs/tracker_config.yml')

    print("Creating Secondary GIE (MobileFaceNet)...")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine")
    sgie.set_property('config-file-path', "configs/sgie_config.txt")

    print("Creating Tiler...")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler_rows = int(math.sqrt(num_sources))
    tiler_columns = int(math.ceil((1.0 * num_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", 1280)
    tiler.set_property("height", 720)

    print("Creating Converter & OSD & Sink...")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    # Using EGL sink for Jetson display
    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    print("Adding elements to Pipeline...")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    if transform:
        pipeline.add(transform)
    pipeline.add(sink)

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

    # Attach Probe to OSD Sink Pad
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write("Unable to get sink pad of nvosd\n")
    else:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Connect to signals
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write(f"Warning: {err}: {debug}\n")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write(f"Error: {err}: {debug}\n")
            loop.quit()
        return True
        
    bus.connect("message", bus_call, loop)

    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except BaseException:
        pass

    print("Exiting app...")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
