from mmcv.runner import HOOKS, Hook


def _set_object_sample_enabled(dataset, enabled):
    # `runner.data_loader.dataset` is the CBGSDataset wrapper, which has no
    # `pipeline_single`; looking for the transform on it directly finds
    # nothing, so the fade silently never happened and pasting ran for the
    # whole schedule. Unwrap to the dataset that actually holds the pipeline.
    depth = 0
    while dataset is not None and not hasattr(dataset, 'pipeline_single'):
        dataset = getattr(dataset, 'dataset', None)
        depth += 1
        if depth > 4:
            return
    pipeline = getattr(dataset, 'pipeline_single', None)
    if pipeline is None:
        return
    for transform in pipeline.transforms:
        if transform.__class__.__name__ == 'TrackConsistentObjectSample':
            transform.enabled = enabled


@HOOKS.register_module()
class ObjectSampleFadeHook(Hook):
    """Disable copy-paste GT sampling for the last `fade_epochs` epochs.

    Mirrors the "fade strategy" TransFusion borrows from PointAugmenting/
    CenterPoint: pasted objects sit in physically implausible spots (no
    occlusion, no consistent ground contact), so training on them for the
    entire schedule teaches the network to expect them at test time. Fading
    them out lets the model re-adjust to the real data distribution before
    training ends.
    """

    def __init__(self, fade_epochs=5):
        self.fade_epochs = fade_epochs

    def before_train_epoch(self, runner):
        dataset = runner.data_loader.dataset
        remaining = runner.max_epochs - runner.epoch
        _set_object_sample_enabled(dataset, enabled=remaining > self.fade_epochs)
