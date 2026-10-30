"""Public, pinned Meta Sapiens v1 Goliath pose exports; no inference imports."""

UPSTREAM_COMMIT = '2cb07227a740cf09896309ea3a3b8fa44429865c'
INFERENCE = 'sapiens-v1-goliath308-coco17-udp-fp32-v1'
INPUT_SIZE = (768, 1024)  # width, height
HEATMAP_SIZE = (192, 256)
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
PADDING = 1.25
BLUR_KERNEL = 11
# Goliath's wrists are in the hand groups, not at COCO indices 9 and 10.
# Verified against pose/configs/_base_/datasets/goliath.py at UPSTREAM_COMMIT.
COCO17_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)
MODELS = {
    '0.3b': dict(repository='facebook/sapiens-pose-0.3b-torchscript',
        revision='796d82b26775444bbeb38a8202f8f9783370603f',
        filename='sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2',
        size=1319574979, sha256='59ea239bfb217caf76cbeac1836460ef8f9ab5f39aeb22f2577373b62fa25edc'),
    '0.6b': dict(repository='facebook/sapiens-pose-0.6b-torchscript',
        revision='fe8cabd29460bea0c046c35f4df10bdc1814061e',
        filename='sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2',
        size=2645835253, sha256='7d5407351b9f20dddea72cfa7be26783835f58d4d565e3aaf4746cccce9b677a'),
    '1b': dict(repository='facebook/sapiens-pose-1b-torchscript',
        revision='9d36f8117f8ad25cb23ada28bf2f7a3b62aaed68',
        filename='sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2',
        size=4677003899, sha256='d1077c15e66fb2798893e569f71cb063f0b0778af2a62e3c0233807a351c31f3'),
}


def contract(variant):
    return dict(variant=variant, **MODELS[variant], inference=INFERENCE,
                source_revision=UPSTREAM_COMMIT, input_size=INPUT_SIZE, heatmap_size=HEATMAP_SIZE,
                mean=MEAN, std=STD, padding=PADDING, blur_kernel=BLUR_KERNEL,
                coco17_indices=COCO17_INDICES)
