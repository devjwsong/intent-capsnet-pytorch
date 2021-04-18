# intent-capsnet-pytorch
This repository is Pytorch implementation of **"Zero-shot User Intent Detection via Capsule Neural Networks"**.

Details of this model is available in the original paper, "*Zero-shot user intent detection via capsule neural networks*"[[1]](#1).

And this Pytorch implementation is revised and upgraded version of the original repository, "*Zero-shot User Intent Detection via Capsule Neural Networks (PyTorch Implementation)*"[[2]](#2).

The details of the model structure are as follows.

<img src="https://user-images.githubusercontent.com/16731987/103389488-81ae0900-4b52-11eb-8220-299d873b5934.PNG" alt="The description of the IntentCapsNet structure."/>

<br/>

---

### Differences

1. Unlike the existing version, you can use BERT[[3]](#3) as the encoder. 

   Also besides word2vec, `nn.Embedding` layer can be added to be trained in the beginning as an option.

   Tokenizers and embedding methods can be different by the model type you choose.

   |               | **bert_capsnet**          | **basic_capsnet**      | **w2v_capsnet**     |
   | ------------- | ------------------------- | ---------------------- | ------------------- |
   | **Encoder**   | BERT(`bert-base-uncased`) | BiLSTM                 | BiLSTM              |
   | **Tokenizer** | BERT Tokenizer            | BERT Tokenizer         | WhiteSpace          |
   | **Embedding** | BERT Embedding            | Pytorch `nn.Embedding` | GoogleNews Word2Vec |

   <br/>

2. In addition to zero shot intent detection task, you can train & test the model in original seen intent classification task.

   All you need to do is just to specify the mode option.

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### Dataset

This repository contains the sample dataset in `data/raw` directory.

The sample is SNIPS NLU benchmark dataset[[4]](#4) parsed only for intent tags and texts.

You can use different dataset but you should set the raw data file same as the sample's format.

Each `txt` file represents one intent and each line in a file consists of intent and text, separated by `\t`.

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### Arguments

| argument                  | type    | description                                                  | default          |
| ------------------------- | ------- | ------------------------------------------------------------ | ---------------- |
| `--seed`                  | `int`   | The random seed.                                             | `0`              |
| `--batch_size`            | `int`   | The batch size.                                              | `16`             |
| `--learning_rate`         | `float` | The learning rate.                                           | `1e-4`           |
| `--num_epochs`            | `int`   | The total number of epochs.                                  | `10`             |
| `--max_len`               | `int`   | The maximum input length.                                    | `128`            |
| `--dropout`               | `float` | The dropout rate.                                            | `0.0`            |
| `--d_a`                   | `int`   | The dimension size of an internal vector during self-attention. | `80`             |
| `--num_props`             | `int`   | The number of properties in each capsule.                    | `10`             |
| `--r`                     | `int`   | The number of semantic features                              | `3`              |
| `--num_iters`             | `int`   | The number of iterations for the dynamic routing algorithm.  | `1`              |
| `--alpha`                 | `float` | The coefficient value for encouraging the discrepancies among different attention heads in the loss function. | `1e-4`           |
| `--sim_scale`             | `int`   | The scaling factor for intent similarity.                    | `1`              |
| `--num_layers`            | `int`   | The number of layers for an LSTM encoder.                    | `1`              |
| `--ckpt_dir`              | `str`   | The directory for trained ckpts.                             | `"saved_models"` |
| `--data_dir`              | `str`   | The directory for data.                                      | `"data"`         |
| `--raw_dir`               | `str`   | The directory for raw data.                                  | `"raw"`          |
| `--train_frac`            | `float` | The ratio of the conversations to be included in the train set. | `0.8`            |
| `--train_prefix`          | `str`   | The train data file name's prefix.                           | `"train"`        |
| `--valid_prefix`          | `str`   | The validation data file name's prefix.                      | `"valid"`        |
| `--model_type`            | `str`   | The model type. (`"bert_capsnet"`, `"basic_capsnet"`, `"w2v_capsnet"`) | `"bert_capsnet"` |
| `--mode`                  | `str`   | Seen class or zero shot? (`"seen_class"`, `"zero_shot"`)     | `"seen_class"`   |
| `--bert_embedding_frozen` | `str`   | Do you want to freeze BERT's embedding layer or not? (`"True"`, `"False"`) | `"False"`        |
| `--gpu`                   | `str`   | The index of gpu to use.                                     | `"0"`            |

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Run below codes to train & test a model. (You might have to adjust each argument as you desire...)

   ```shell
   sh exec_train.sh
   ```

   You will have the processed data files as follows. (If you follow the default directory names...)

   ```
   data
   └--raw
   	└--intent0.txt
   	└--intent1.txt
   	└--...
   	└--intent(I-1).txt
   └--MODE(seen_class/zero_shot)
   	└--train.txt
   	└--valid.txt
   ```

   <br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### References

<a id="1">[1]</a> 
*Xia, C., Zhang, C., Yan, X., Chang, Y., & Yu, P. S. (2018). Zero-shot user intent detection via capsule neural networks. arXiv preprint arXiv:1809.00385*. ([https://arxiv.org/abs/1809.00385](https://arxiv.org/abs/1809.00385))

<a id="2">[2]</a> 
*Zero-shot User Intent Detection via Capsule Neural Networks (PyTorch Implementation)*. ([https://github.com/nhhoang96/ZeroShotCapsule-PyTorch-](https://github.com/nhhoang96/ZeroShotCapsule-PyTorch-))

<a id="3">[3]</a> 
*Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.*

<a id="4">[4]</a> 
*Natural Language Understanding benchmark.* ([https://github.com/sonos/nlu-benchmark](https://github.com/sonos/nlu-benchmark))
