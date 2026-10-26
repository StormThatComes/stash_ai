"""Pinned upstream SapiensID assets; importable without inference dependencies."""
UPSTREAM_COMMIT = 'a8d8011117db23f20c7ade3bc38e8075810ac886'
SOURCE_URL = 'https://codeload.github.com/mk-minchul/sapiensid/zip/' + UPSTREAM_COMMIT
SOURCE_BYTES = 52297316
SOURCE_SHA256 = '886e39649e729d22a83b7a5e829fc9dc2c91f3a85cb3b2347a509f528e8bf62f'
SOURCE_TREE_SHA256 = '0a128079b35da7975c332cdf86b7b7fa54159d8dad8ad2b14a71c3c961854dab'
# Pose checkpoint shipped in that same upstream revision (not a floating release).
POSE_URL = ('https://raw.githubusercontent.com/mk-minchul/sapiensid/' + UPSTREAM_COMMIT +
            '/tasks/WebBody/c_detect/yolov8n-pose.pt')
POSE_BYTES = 6832633
POSE_SHA256 = 'c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212'
MODEL_YAML_SHA256 = '2aa66245bdf81631cf243aff26dc2bca30788ea77958981d1947bd750cbf93b9'
# Official pretrained archive linked from tasks/sapiensID/README.md at UPSTREAM_COMMIT.
CHECKPOINT_URL = ('https://drive.usercontent.google.com/download?'
                  'id=18d8Ogw60zxnaIIjSb99Y8oLvzCEn_L-Z&export=download&confirm=t')
CHECKPOINT_BYTES = 3053586112
WB4M_BYTES = 1646175550
WB4M_SHA256 = '8df1e2d7a64fbc87969326862aeafccc83eb5ee0462a4f007d7f09603782cf02'
# Raw DEFLATE entry offsets/lengths, verified from the pinned ZIP central/local headers.
# Expanded checkpoint hashes were verified against the original upstream archive.
# Both variants share the exact model.yaml bytes, but their weights differ.
MODELS = {
    'wb4m': dict(size=WB4M_BYTES, sha256=WB4M_SHA256,
                 ranges={'model.pth': (249, 1526109158), 'model.yaml': (1526109502, 549)}),
    'wb12m': dict(size=1646175550,
                  sha256='b91db9d4ea13542b1bff76deaa0d678b9586c73506c7ae0116ab907f688a1a60',
                  ranges={'model.pth': (1526110232, 1527474508), 'model.yaml': (3053584836, 549)}),
}
