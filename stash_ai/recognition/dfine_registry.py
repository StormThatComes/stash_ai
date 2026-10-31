"""Pinned Objects365 -> COCO D-FINE person detectors; no inference imports."""

INFERENCE = 'dfine-rgb640-fp32-person-before-topk-cpu-nms-v1'
PERSON_LABEL = 0
MODELS = {
    'dfine_l': dict(
        repository='ustc-community/dfine-large-obj2coco-e25',
        revision='c89de47153e338c07cddc81fc21a776b97cbf75e',
        assets={
            'config.json': (6973, 'b3cc9b27711d161f6b6fa9b5c6a8a7417403113ffe45e5f358829fbe1c15d206'),
            'preprocessor_config.json': (470, '7e9e2245806aa3ef068a5321f5eebee3116fa4cec560173231954ffaca930eca'),
            'model.safetensors': (125618504, '82508394c2070c7d82c286b78f8b4b42dd64b5a3144578427c6e67b07b9f8154'),
        }),
    'dfine_x': dict(
        repository='ustc-community/dfine-xlarge-obj2coco',
        revision='15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d',
        assets={
            'config.json': (6974, 'cad171dc7c4e51891f57db3fdee3d47b68ca73a786b61fddbeeb63ccc883e5b2'),
            'preprocessor_config.json': (470, '7e9e2245806aa3ef068a5321f5eebee3116fa4cec560173231954ffaca930eca'),
            'model.safetensors': (251619400, '19246b6d70c175d552752c8b7fac9b37516d8b14df2476a2cfd174837fc4df37'),
        }),
}


def contract(detector):
    model = MODELS[detector]
    return dict(detector=detector, repository=model['repository'], revision=model['revision'],
                inference=INFERENCE, person_label=PERSON_LABEL,
                assets={name: checksum for name, (_, checksum) in model['assets'].items()})
