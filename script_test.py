import time
from pathlib import Path
from string import Template
import depthai.node as n
from depthai import Pipeline, Device, OpenVINO

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PRODUCER = str(
    SCRIPT_DIR / "pipeline_scripts" / "tests" / "producer_callback_test.py"
)
SCRIPT_CONSUMER = str(
    SCRIPT_DIR / "pipeline_scripts" / "tests" / "slow_consumer.py"
)

if __name__ == "__main__":
    def build_manager_script(script_path, pipeline):
        """
        The code of the scripting node 'manager_script' depends on:
            - the score threshold,
            - the video frame shape,
        So we build this code from the content of the file template_manager_script_*.py,
        which is a python template
        """
        manager_script = pipeline.create(n.Script)

        # Read the template
        with open(
                script_path,
                "r",
        ) as file:
            template = Template(file.read())
            name = file.name.split("/")[-1].split(".")[0]
            name += (" " * (15 - len(name))) + ":"

        subs = {
            "_STUB_IMPORTS": '"""',
            "_TRACE1": "#",
            "_TRACE2": "node.warn",
            "_fps": 10,
            "_pd_score_thresh": 0,
            "_lm_score_thresh": 0,
            "_pad_h": 0,
            "_img_h": 0,
            "_img_w": 0,
            "_frame_size": 0,
            "_crop_w": 0,
            "_body_pre_focusing": 0,
            "_body_score_thresh": 0,
            "_body_input_length": 0,
            "_hands_up_only": 0,
            "_single_hand_tolerance_thresh": 0,
        }
        # Perform the substitution
        code = template.substitute(**subs)
        # Remove comments and empty lines
        import re

        code = re.sub(r'"{3}.*?"{3}', "", code, flags=re.DOTALL)
        # Remove None placeholders
        code = re.sub(r"[^ ]+ {2}###", "", code)
        # Remove triple comment on traces and blocks
        code = re.sub(r"###", "", code)
        code = re.sub(r"#.*", "", code)
        code = re.sub('\n\s*\n', "\n", code)

        manager_script.setScript(code, name)
        return manager_script

    device = Device()
    pipeline: Pipeline = Pipeline()
    pipeline.setOpenVINOVersion(version=OpenVINO.Version.VERSION_2022_1)

    producer_script = build_manager_script(SCRIPT_PRODUCER, pipeline)
    consumer_script = build_manager_script(SCRIPT_CONSUMER, pipeline)

    producer_script.outputs["from_producer"].link(consumer_script.inputs["from_producer"])
    consumer_script.inputs["from_producer"].setBlocking(False)
    consumer_script.inputs["from_producer"].setQueueSize(2)

    device.startPipeline(pipeline)

    time.sleep(10)

    device.close()