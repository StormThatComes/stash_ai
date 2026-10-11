# Third-party notices

The original Stash AI project source code is licensed under the
[GNU General Public License v3.0 only](LICENSE) (`GPL-3.0-only`).

**That license does not relicense third-party models, model weights,
upstream source code, or installed dependencies.** Users must comply
with the terms supplied by each respective upstream project.

## Models downloaded at runtime

Stash AI downloads model artifacts from Hugging Face at pinned revisions.
Model weights are not committed to this repository.

| Model | Upstream author/repository | License declared upstream | Revision used by Stash AI |
| --- | --- | --- | --- |
| JoyTag | [fancyfeast/joytag](https://huggingface.co/fancyfeast/joytag) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | `6b7f16331a6ccf0fdce37d5a9564715f6e772b22` |
| WD EVA02-Large Tagger v3 | [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | `b25b82a03f7282e41aa2f257a52c7583b710bd1c` |
| DanbooruTagQuery L/16 | [realphongha/danbooru-tag-query](https://huggingface.co/realphongha/danbooru-tag-query) | [DINOv3 License](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/blob/main/LICENSE.md) (`dinov3-license`, **not** GPL or Apache) | `d2f8da27a23db4adf95c3e663af183b92784ad3e` |
| OppaiOracle V1.1 | [Grio43/OppaiOracle](https://huggingface.co/Grio43/OppaiOracle) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | `96992fa30568c386e9fe7c8a1a68f798a3202c09` |
| YuNet face detector | [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet) | MIT | `3cc26e7f1014a5ee5d74a42acee58bafc9d0a310` |
| SFace face embeddings | [opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface) | Apache-2.0 | `3d7082438a6e4551e840c9b2bb60b71e8da4b524` |

The upstream license information above describes the referenced model
repositories and should be rechecked before redistributing model assets.
In particular, the DINOv3 License imposes separate terms on the
DanbooruTagQuery model. Publishing this project's GPL-licensed code
does **not** grant permissions beyond those upstream terms.

## JoyTag architecture loader

Stash AI also downloads `Models.py` from the upstream
[fancyfeast/joytag Hugging Face Space](https://huggingface.co/spaces/fancyfeast/joytag)
at revision `bce35c9afc4dbe5eb71ee836939c84d4614eec96`.
The upstream Space declares Apache-2.0. That source file is not
vendored in this repository; its upstream license applies.

## Other third-party software

Python packages listed in `requirements*.txt` and Stash itself are
separate projects with their own licenses. For example,
[Stash](https://github.com/stashapp/stash) is licensed under
[AGPL-3.0](https://github.com/stashapp/stash/blob/develop/LICENSE).
This repository's GPL license does not replace their licenses.

This notice provides attribution and identifies relevant external
licenses; it is not a substitute for reading the current license text
and notices provided by each upstream project.
