from __future__ import annotations

from typing import Any
import logging

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy
from .control.safety import clamp_action_chunk


class PolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        image_size: int | None = 224,
        joint_limit_margin: float = 0.01,
    ) -> None:
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._image_size = image_size
        self._joint_limit_margin = joint_limit_margin
        self._rtc_mask_logged = False

    def prepare_images(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        prepared = {}
        for name in ("left_wrist", "right_wrist", "third_person"):
            image = np.asarray(images[name])
            if self._image_size is not None:
                image = image_tools.resize_with_pad(image, self._image_size, self._image_size)
            image = image_tools.convert_to_uint8(image)
            prepared[name] = np.ascontiguousarray(image)
        return prepared

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = dict(observation)
        observation["images"] = self.prepare_images(observation["images"])
        observation["state"] = np.asarray(observation["state"], dtype=np.float32)
        rtc_prev_actions = observation.get("rtc_prev_actions")
        if rtc_prev_actions is not None:
            observation["rtc_prev_actions"] = np.asarray(rtc_prev_actions, dtype=np.float32)
        # Surface RTC/model errors instead of silently changing the inference mode.
        # policy.rtc.enabled=false explicitly selects unguided async inference.
        result = self._client.infer(observation)
        if rtc_prev_actions is not None:
            expected = [*range(7), *range(8, 15)]
            if result.get("rtc_guided_action_indices") != expected:
                raise RuntimeError("Server has not confirmed joints-only Nero RTC. Restart the updated policy server.")
            if not self._rtc_mask_logged:
                logging.getLogger(__name__).info("RTC guided action indices=%s; gripper and padding direct guidance excluded", expected)
                self._rtc_mask_logged = True
        actions = np.asarray(result.get("actions"))
        if actions.ndim == 2 and actions.shape[1] == 16:
            actions, clipped, max_clip = clamp_action_chunk(actions, self._joint_limit_margin)
            result["actions"] = actions
            result["safety_clipped_values"] = clipped
            result["safety_max_clip"] = max_clip
        return result

    def reset(self) -> None:
        self._client.reset()

    def metadata(self) -> dict[str, Any]:
        return self._client.get_server_metadata()
