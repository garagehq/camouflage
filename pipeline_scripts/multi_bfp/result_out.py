import marshal
import time

###${_STUB_IMPORTS} # noqa
from ..import_stub import *
from ..import_stub import node
###${_STUB_IMPORTS} # noqa

script_name = ""  ###${_NAME} # noqa
fps = 0.0  ###${_fps}# noqa

frame_queue_size = 0  ###${_frame_queue_size}# noqa
lm_score_thresh = 0.0  ###${_lm_score_thresh}# noqa


class FrameQueue:
    def __init__(self, first_seq_no) -> None:
        self.queue = [None] * frame_queue_size
        self.frame_no = first_seq_no

    def force_send_or_drop(self):
        message = self.queue.pop(0)
        if message is not None:
            # noinspection PyArgumentList
            send_message(*message)
        else:
            ###${_TRACE_INFO}node.warn(f"{script_name} dropping frame: {self.frame_no}")${_TRACE_INFO} # noqa
            del message
        self.queue.append(None)
        self.frame_no += 1

    def push(self, seq_no, data):
        ###${_TRACE_INFO} # noqa
        # noinspection PyUnusedLocal
        before = str(self.queue)
        ###${_TRACE_INFO} # noqa
        insert_frame_no = seq_no
        ###${_TRACE_INFO}(f"{script_name} inserting {insert_frame_no}")${_TRACE_INFO} # noqa
        idx_diff = insert_frame_no - self.frame_no
        if idx_diff > len(self.queue) - 1:
            for i in range(idx_diff - len(self.queue) - 1):
                self.force_send_or_drop()
            self.queue[-1] = data
        else:
            self.queue[idx_diff] = data
        ###${_TRACE_INFO}node.warn(f"{script_name} before: {before} after: {self.queue}")${_TRACE_INFO} # noqa

    def send_all_in_sequence(self):
        while self.queue[0] is not None:
            self.force_send_or_drop()


# noinspection PyArgumentList,PyTypeChecker
def send_message(frame, detection_data):
    result_serial = marshal.dumps(detection_data)
    buffer = Buffer(len(result_serial))
    buffer.setData(result_serial)
    node.io['results_out'].send(buffer)
    node.io['results_out'].send(frame)
    ###${_TRACE2}node.warn("Result manager sent result to host")${_TRACE2} # noqa


frame_queue = None

early_output_queue_values = {
    "early_out_pd": {"pd_inf": False, "nb_lm_inf": False},
    "early_out_lm": {"pd_inf": True, "nb_lm_inf": False}
}

time.sleep(1 / fps)

while True:
    ###${_TRACE_INFO}node.warn(f"{script_name} waking")${_TRACE_INFO} # noqa

    new_frames = {q: node.io[q].tryGet() for q in early_output_queue_values}
    if any(new_frames.values()):
        # Purge None vals
        new_frames = {key: val for key, val in new_frames.items() if val is not None}
        if frame_queue is None:
            first_frame_num = new_frames[min(new_frames, key=lambda k: new_frames[k].getSequenceNum())].getSequenceNum()
            ###${_TRACE_INFO}node.warn(f"{script_name} init first frame{first_frame.getSequenceNum()}")${_TRACE_INFO} # noqa
            frame_queue = FrameQueue(first_frame_num-frame_queue_size if first_frame_num >= frame_queue_size else 0)
        for q_key in new_frames:
            f_no = new_frames[q_key].getSequenceNum()
            frame_queue.push(f_no, [new_frames[q_key], early_output_queue_values[q_key]])
    if frame_queue is not None:
        frame_queue.send_all_in_sequence()
    # sleep for one third of a frame
    ###${_TRACE_INFO}node.warn(f"{script_name} finished processing {len([val for val in new_frames.values() if val is not None])} frames")${_TRACE_INFO} # noqa
    time.sleep(.33 / fps)
