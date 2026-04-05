from comfy_api.latest import ComfyExtension
from .nodes import VCGLoadModel, VCGGenerateLUT, VCGApplyLUT


class VCGExtension(ComfyExtension):
    async def get_node_list(self):
        return [VCGLoadModel, VCGGenerateLUT, VCGApplyLUT]


async def comfy_entrypoint() -> VCGExtension:
    return VCGExtension()
