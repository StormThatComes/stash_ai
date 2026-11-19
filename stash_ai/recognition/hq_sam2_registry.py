"""Verified upstream HQ-SAM 2 Large inference sources and checkpoint."""
SOURCE_REVISION = 'e696978d60352dc9a26b12631cd91781502c6546'
SOURCE_URL = f"https://raw.githubusercontent.com/SysCV/sam-hq/{SOURCE_REVISION}/"
REVISION = "09b02a333b37772133eff3997bdba997867374b7"
NAME = "sam2.1_hq_hiera_large.pt"
SIZE = 898844313
SHA256 = "03b8be4b4f282c3d7365715a033918d9ac67cbc597660ccc274448aa1c8cdb05"
URL = f"https://huggingface.co/lkeab/hq-sam/resolve/{REVISION}/{NAME}"
CONFIG = "sam-hq2/sam2/configs/sam2.1/sam2.1_hq_hiera_l.yaml"
SOURCE_FILES = {'LICENSE': {'sha256': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4', 'size': 11357},
 'sam-hq2/sam2/configs/sam2.1/sam2.1_hq_hiera_l.yaml': {'sha256': 'a51bde6c47d8c76b49c3b7b2213c382b96f9827ca1f55ce4c4fb8096a134df5a',
                                                        'size': 3803},
 'sam-hq2/sam2/modeling/backbones/hieradet.py': {'sha256': '03785581ca304d0451ae0df7a08ee0bf1e1dbe66fad285066e6b9ffc0d88d64f',
                                                 'size': 10003},
 'sam-hq2/sam2/modeling/backbones/image_encoder.py': {'sha256': '16eaad232220386f510ca8dfab8655e63699977149a4daf9ce9c6f374e6777a9',
                                                      'size': 4706},
 'sam-hq2/sam2/modeling/backbones/utils.py': {'sha256': '84bd48cda1ad468fb8c231d0c2eb83ad53386633d6e63ddaa6ed929c5521bdba',
                                              'size': 3053},
 'sam-hq2/sam2/modeling/memory_attention.py': {'sha256': '07358cb7f58ec3788e88ff4e6415f8a9d628a3963529d6e2c915494a5347c8e2',
                                               'size': 5509},
 'sam-hq2/sam2/modeling/memory_encoder.py': {'sha256': '73f7089ae5fdacdfcaaf3deca1ca6d9f84d89e7deb9cf97589622a9ef17ba42f',
                                             'size': 5657},
 'sam-hq2/sam2/modeling/position_encoding.py': {'sha256': '14ae89d7ae68f61e2ffcba09eb171d8df9a7298332d4da99036d703294f89ec1',
                                                'size': 8361},
 'sam-hq2/sam2/modeling/sam/mask_decoder.py': {'sha256': 'ca3523c58365574faddf1bfb54f374e4b4beba05127185ea2b17d22b916e099a',
                                               'size': 12657},
 'sam-hq2/sam2/modeling/sam/mask_hq_decoder.py': {'sha256': 'e9ecebb403219242fe68d08298733d5b7adde9ab16d15e852ab6bd212424255f',
                                                  'size': 14149},
 'sam-hq2/sam2/modeling/sam/prompt_encoder.py': {'sha256': '486596f5ba976cb9ced3836d65f1793dae586be2180567a451098f96c45ae401',
                                                 'size': 7016},
 'sam-hq2/sam2/modeling/sam/transformer.py': {'sha256': '17aac13abc8f73023f6be4b78af708df9f9f254964729421b1ba60e72a9011c1',
                                              'size': 12870},
 'sam-hq2/sam2/modeling/sam2_base.py': {'sha256': '6d81450e897d0735f9be369771f2b3fb6eadb90dd3f3ac16b3f7a8c8eb1a052a',
                                        'size': 46907},
 'sam-hq2/sam2/modeling/sam2_hq_base.py': {'sha256': 'be6f6d6799d85c38a29d5e385de03cde0a58342d947727001acb34dcec6eff57',
                                           'size': 46916},
 'sam-hq2/sam2/modeling/sam2_utils.py': {'sha256': 'e35bbf13bc2e544a0272cd3f9539af38a752676ac3cd744be31cd8220afee804',
                                         'size': 13173},
 'sam-hq2/sam2/sam2_image_predictor.py': {'sha256': 'e9fa4bbac3a2076e755f29eab2d1d8679c32bb148e485c997c532519a24e2e49',
                                          'size': 20217},
 'sam-hq2/sam2/utils/misc.py': {'sha256': '01600c01c161cd079d7106fb1d4da845cf91aa31ab2bcecaf8cb151b6d6d20a2',
                                'size': 13090},
 'sam-hq2/sam2/utils/transforms.py': {'sha256': 'ba3a64f4600c62f209206a6df3b40e3fcf133edae32fad658831bb0c2a6d1146',
                                      'size': 4885}}
