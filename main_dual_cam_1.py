import gi
import sys
import ctypes
import math
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

from db_utils import load_all_embeddings

# ----------------- GLOBALS ----------------- #
embeddings_db = load_all_embeddings()
SIMILARITY_THRESHOLD = 0.40
MUXER_BATCH_TIMEOUT_USEC = 40000

frame_count_map = {}

# ----------------- UTILS ----------------- #

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------- PROBE ----------------- #

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

        stream_id = frame_meta.pad_index
        frame_count_map.setdefault(stream_id, 0)
        frame_count_map[stream_id] += 1

        frame_number = frame_count_map[stream_id]

        print(f"[Stream {stream_id}] Frame {frame_number} | Faces: {frame_meta.num_obj_meta}")

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

            obj_meta.rect_params.border_width = 3
            obj_meta.rect_params.border_color.set(1, 0, 0, 1)
            obj_meta.text_params.display_text = "Detecting..."

            l_user_meta = obj_meta.obj_user_meta_list

            while l_user_meta is not None:
                user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)

                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVDSINFER_TENSOR_OUTPUT_META"):
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    v = np.ctypeslib.as_array(ptr, shape=(layer.inferDims.d[0],))

                    emb = np.copy(v)
                    norm = np.linalg.norm(emb)

                    if norm > 0:
                        emb /= norm

                        best_match = None
                        best_score = -1

                        for user in embeddings_db:
                            score = cosine_similarity(emb, user['embedding'])
                            if score > best_score:
                                best_score = score
                                best_match = user

                        if best_score > SIMILARITY_THRESHOLD:
                            obj_meta.text_params.display_text = f"{best_match['name']} ({best_score:.2f})"
                            obj_meta.rect_params.border_color.set(0, 1, 0, 1)

                l_user_meta = l_user_meta.next

            l_obj = l_obj.next

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

# ----------------- SOURCE ----------------- #

def create_source_bin(index, uri):
    bin_name = f"source-bin-{index}"
    nbin = Gst.Bin.new(bin_name)

    v4l2src = Gst.ElementFactory.make("v4l2src", f"v4l2src_{index}")
    v4l2src.set_property("device", uri)

    caps1 = Gst.ElementFactory.make("capsfilter", f"caps1_{index}")
    caps1.set_property("caps",
        Gst.Caps.from_string("video/x-raw, width=640, height=480, framerate=30/1")
    )

    vidconv = Gst.ElementFactory.make("videoconvert", f"vidconv_{index}")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv_{index}")

    caps2 = Gst.ElementFactory.make("capsfilter", f"caps2_{index}")
    caps2.set_property("caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    )

    Gst.Bin.add(nbin, v4l2src)
    Gst.Bin.add(nbin, caps1)
    Gst.Bin.add(nbin, vidconv)
    Gst.Bin.add(nbin, nvvidconv)
    Gst.Bin.add(nbin, caps2)

    v4l2src.link(caps1)
    caps1.link(vidconv)
    vidconv.link(nvvidconv)
    nvvidconv.link(caps2)

    ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    nbin.add_pad(ghost)
    ghost.set_target(caps2.get_static_pad("src"))

    return nbin

# ----------------- MAIN ----------------- #

def main(args):

    if len(args) < 2:
        print("Usage: python3 main.py /dev/video0")
        sys.exit(1)

    Gst.init(None)
    pipeline = Gst.Pipeline()

    source_bin = create_source_bin(0, args[1])

    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    streammux.set_property("width", 640)
    streammux.set_property("height", 480)
    streammux.set_property("batch-size", 1)
    streammux.set_property("live-source", 1)

    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", "configs/pgie_config.txt")

    sgie = Gst.ElementFactory.make("nvinfer", "sgie")
    sgie.set_property("config-file-path", "configs/sgie_config.txt")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "conv")
    nvosd = Gst.ElementFactory.make("nvdsosd", "osd")

    transform = Gst.ElementFactory.make("nvegltransform", "transform")
    sink = Gst.ElementFactory.make("nveglglessink", "sink")

    pipeline.add(source_bin)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(transform)
    pipeline.add(sink)

    srcpad = source_bin.get_static_pad("src")
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(sgie)
    sgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(transform)
    transform.link(sink)

    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    loop = GLib.MainLoop()
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
