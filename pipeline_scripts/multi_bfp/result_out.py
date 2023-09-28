import marshal
import time

###${_STUB_IMPORTS} # noqa
from depthai import *
###${_STUB_IMPORTS} # noqa

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
            ###${_TRACE_DUMP}(f"${_NAME} dropping frame: {self.frame_no}") # noqa
            del message
        self.queue.append(None)
        self.frame_no += 1

    # noinspection PyUnresolvedReferences
    def push(self, seq_no, data):
        before = str(self.queue)
        insert_frame_no = seq_no
        ###${_TRACE_DUMP}(f"${_NAME} inserting {insert_frame_no}")
        idx_diff = insert_frame_no - self.frame_no
        if idx_diff > len(self.queue) - 1:
            for i in range(idx_diff - len(self.queue) - 1):
                self.force_send_or_drop()
            self.queue[-1] = data
        else:
            self.queue[idx_diff] = data
        ###${_TRACE_DUMP}(f"${_NAME} before: {before} after: {self.queue}")

    def send_all_in_sequence(self):
        while self.queue[0] is not None:
            self.force_send_or_drop()


# noinspection PyArgumentList,PyTypeChecker,PyUnresolvedReferences
def send_message(frame, detection_data):
    result_serial = marshal.dumps(detection_data)
    buffer = Buffer(len(result_serial))
    buffer.setData(result_serial)
    node.io['host'].send(buffer)
    node.io['host'].send(frame)
    ###${_TRACE2}("Result manager sent result to host") # noqa


frame_queue = None

early_output_queue_values = {
    "early_out_pd": {"pd_inf": False, "nb_lm_inf": False},
    "early_out_lm": {"pd_inf": True, "nb_lm_inf": False}
}

time.sleep(1 / fps)

while True:
    ###${_TRACE_DUMP}(f"${_NAME} waking") # noqa

    # noinspection PyUnresolvedReferences
    new_frames = {q: node.io[q].tryGet() for q in early_output_queue_values}
    if any(new_frames.values()):
        # Purge None vals
        new_frames = {key: val for key, val in new_frames if val is not None}
        if frame_queue is None:
            first_frame_numeric = new_frames[min(new_frames, key=lambda k: new_frames[k].getSequenceNum())]
            frame_queue = FrameQueue(first_frame_numeric-frame_queue_size)
        for q_key in new_frames:
            f_no = new_frames[q_key].getSequenceNum()
            frame_queue.push(f_no, [new_frames[q_key], early_output_queue_values[q_key]])
    if frame_queue is not None:
        frame_queue.send_all_in_sequence()
    # sleep for one third of a frame
    ###${_TRACE_DUMP}(f"${_NAME} finished processing {len([val for val in new_frames.values() if val is not None])} frames") # noqa
    time.sleep(.33 / fps)
