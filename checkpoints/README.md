# TabPFN checkpoint

Pretrained TabPFN-2.6 weights are not committed to this MIT-licensed repository. Obtain them under Prior Labs' model terms:

```bash
bibm-replicate fetch-checkpoint --destination checkpoints
bibm-replicate verify-checkpoint checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt
```

Expected SHA-256:

```text
0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a
```

The filename is ignored by Git through the repository's `*.ckpt` rule.
