"""ARX OpenPI tube/human-v2 LeRobot registry."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class ArxOpenPiDeltaPose7DDataConfig:
    action_type = "delta_ee"
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    gripper_action_key = "action.delta_pos.gripper"
    gripper_action_threshold = 1.3
    gripper_action_low = 0.0
    gripper_action_high = 3.45
    video_keys = ["video.front", "video.wrist"]
    state_keys = [
        "state.end_effector_pos.x",
        "state.end_effector_pos.y",
        "state.end_effector_pos.z",
        "state.end_effector_pos.roll",
        "state.end_effector_pos.pitch",
        "state.end_effector_pos.yaw",
        "state.gripper.pos",
    ]
    action_keys = [
        "action.delta_pos.x",
        "action.delta_pos.y",
        "action.delta_pos.z",
        "action.delta_pos.roll",
        "action.delta_pos.pitch",
        "action.delta_pos.yaw",
        "action.delta_pos.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(32))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        state_modes = {key: "q99" for key in self.state_keys}
        action_modes = {key: "q99" for key in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "arx_openpi_deltapose_7d": ArxOpenPiDeltaPose7DDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "arx_openpi_deltapose_7d": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "ARX_openpi_tube_human_v2": [
        (
            "/mnt/inspurfs/vla_coop/data_2/haoran_data/zsp_zhr_hfx_puttube_merged_0421",
            1.0,
            "arx_openpi_deltapose_7d",
        ),
        (
            "/mnt/inspurfs/vla_coop/data_2/haoran_data/testtube-0427-0428-whole-recovery",
            1.0,
            "arx_openpi_deltapose_7d",
        ),
        (
            "/mnt/inspurfs/vla_coop/data_2/haoran_data/human_v2",
            1.0,
            "arx_openpi_deltapose_7d",
        ),
    ],
}
