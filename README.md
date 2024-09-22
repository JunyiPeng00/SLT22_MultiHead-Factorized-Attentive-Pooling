# Wespeaker implementation
I re-wrote the whole system with wespeaker toolkit and achieved higher results, which can be found [here](https://github.com/BUTSpeechFIT/wespeaker_ssl_public/tree/hubert/examples/voxceleb/v4)
| Model | AS-Norm | LMFT | QMF | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------------------|:-------:|:---|:---:|:------------:|:------------:|:------------:|
| WavLM Base Plus + MHFA            | √ | × | × | 0.750 | 0.716 | 1.442 |
| WavLM Large + MHFA            | √ | × | × | 0.649 | 0.610 | 1.235 |

# SLT22_MultiHead-Factorized-Attentive-Pooling

This repository contains the Pytorch code of our paper titled as [An attention-based backend allowing efficient fine-tuning of transformer models for speaker verification](https://arxiv.org/abs/2210.01273). This implementation is based on[ vox_trainer](https://github.com/clovaai/voxceleb_trainer).

---

## Dependencies & Data preparation

- Pls follow [Data preparation](https://github.com/clovaai/voxceleb_trainer#data-preparation) for voxceleb, musan, and RIR datasets.
- Download the pre-trained model (e.g. [WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm)) from huggingface or its official website.
- Modify the corresponding `train_list`, `test_list`, `train_path`, `test_path`, `musan_path`, `rir_path` in `trainSpeakerNet.py` and `trainSpeakerNet_Eval.py`, respectively.
- Add absolute path to the pre-trained model to `pretrained_model_path` in both `yaml/Baseline.yaml` and `yaml/Baseline_lm.yaml`

## Training and Testing

```
name=Baseline.yaml
name_lm=Baseline_lm.yaml

python3 trainSpeakerNet.py --config yaml/$name --distributed >> log/$name.log
python3 trainSpeakerNet.py --config yaml/$name_lm --distributed >> log/$name_lm.log
python3 trainSpeakerNet_Eval.py --config yaml/$name_lm  --eval >> log/$name_lm.log
```

where `lm` denotes large margin fine-tuning.


## Citation

If you used this code, please kindly consider citing the following paper:

```shell notranslate position-relative overflow-auto
@INPROCEEDINGS{10022775,
  author={Peng, Junyi and Plchot, Oldřich and Stafylakis, Themos and Mošner, Ladislav and Burget, Lukáš and Černocký, Jan},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={An Attention-Based Backend Allowing Efficient Fine-Tuning of Transformer Models for Speaker Verification}, 
  year={2023},
  volume={},
  number={},
  pages={555-562},
  doi={10.1109/SLT54892.2023.10022775}}
```

