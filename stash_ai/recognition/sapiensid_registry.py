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
# Raw DEFLATE entry offsets/lengths, verified from the pinned ZIP central/local headers.
# Download wb4m only; never fetch or deserialize wb12m. Expanded bytes are SHA256-checked.
CHECKPOINT_RANGES = {'model.pth': (249, 1526109158), 'model.yaml': (1526109502, 549)}
