import diffusers
from modules.sd_samplers_common import SamplerData

samplers = [
    SamplerData("pndm", diffusers.PNDMScheduler, [], {}),
    SamplerData("lms", diffusers.LMSDiscreteScheduler, [], {}),
    SamplerData("heun", diffusers.HeunDiscreteScheduler, [], {}),
    SamplerData("ddim", diffusers.DDIMScheduler, [], {}),
    SamplerData("ddpm", diffusers.DDPMScheduler, [], {}),
    SamplerData("euler", diffusers.EulerDiscreteScheduler, [], {}),
    SamplerData("euler-ancestral", diffusers.EulerAncestralDiscreteScheduler, [], {}),
    SamplerData("dpm", diffusers.DPMSolverMultistepScheduler, [], {}),
]
