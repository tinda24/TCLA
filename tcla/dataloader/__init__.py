__all__ = [
    "RoboTwinAEWFreqNormDataset",
    "RoboTwinPolicyDataset",
]


def __getattr__(name):
    if name == "RoboTwinAEWFreqNormDataset":
        from tcla.dataloader.robotwin_ae_w_freqnorm import RoboTwinAEWFreqNormDataset

        return RoboTwinAEWFreqNormDataset
    if name == "RoboTwinPolicyDataset":
        from tcla.dataloader.robotwin import RoboTwinPolicyDataset

        return RoboTwinPolicyDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
