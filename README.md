# intent-capsnet-kor-pytorch
This repository is Pytorch implementation of **"Zero-shot User Intent Detection via Capsule Neural Networks"**, especially for Korean.



Details of this model is available in the original paper, [*Xia, C., Zhang, C., Yan, X., Chang, Y., & Yu, P. S. (2018). Zero-shot user intent detection via capsule neural networks. arXiv preprint arXiv:1809.00385*](https://arxiv.org/abs/1809.00385).

And this Pytorch implementation is revised and upgraded version of the original repository, [*Zero-shot User Intent Detection via Capsule Neural Networks (PyTorch Implementation)*](https://github.com/nhhoang96/ZeroShotCapsule-PyTorch-).

<br/>

---

### Differences

1. Unlike the existing version, you can use DistilKoBERT as the encoder. 

   Also besides w2v, `nn.Embedding` layer can be added to be trained in the beginning as an option.

   Tokenizers and embedding methods can be different by the model type you choose.

   |               | **bert_capsnet**       | **basic_capsnet**    | **w2v_capsnet** |
   | ------------- | ---------------------- | -------------------- | --------------- |
   | **Encoder**   | DistilKoBERT           | BiLSTM               | BiLSTM          |
   | **Tokenizer** | KoBERT Tokenizer       | KoBERT Tokenizer     | WhiteSpace      |
   | **Embedding** | DistilKoBERT Embedding | Pytorch nn.Embedding | Korean Word2Vec |

   <br/>

2. In addition to zero shot intent detection task, you can train & test the model in original seen intent classification task.

   All you need to do is just to specify the mode option.

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Run below codes to train & test a model.

   ```shell
   python src/main.py --model_type=MODEL_TYPE --mode=MODE --bert_embedding_frozen=TRUE_OR_FALSE
   ```

   - `--model_type`: You should select one model type among three, `bert_capsnet`, `basic_capsnet`, `w2v_capsnet`.
   - `--mode`: You should choose one of two tasks, `seen_class` or `zero_shot`.
   - `--bert_embedding_frozen`: This matters when you use `bert_capsnet`, which specify whether the embedding layer of DistilKoBERT should be frozen or not. This parameter is `True` or `False` and if you omit this, it is fixed in `False`.

<br/>

