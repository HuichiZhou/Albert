# 文本分类

文本分类任务的主要思想是将文本喂入分类器，分类器进行输出分类结果。  
其在现实中的运用有 垃圾邮件过滤、语义分析、话题检测等。  

## 定义输入和输出
在AllenNLP中，每一个训练例子由Instance对象构成，每一个Instance又由>=1个Field构成。  

## 读取数据  
在movie_review文件夹的tsv文件中，文本的格式由text +[tab]+ label 构成。  
我们需要构造函数读取tsv/csv文件因此设计ClassificationTsvReader 
```python

class ClassificationTsvReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {"text": text_field}
        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)

    # def _read(self, file_path: str) -> Iterable[Instance]:
    #     with open(file_path, "r") as lines:
    #         for line in lines:
    #             text, sentiment = line.strip().split("\t")
    #             yield self.text_to_instance(text, sentiment)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=" ")  # 使用空格作为分隔符，可根据实际情况修改
            for row in reader:
                sentiment = row[0]  # 情感标签在第一列
                text = " ".join(row[1:])  # 将剩余的单词合并为文本
                yield self.text_to_instance(text, sentiment)
```

## 设计分类任务框架
在上一步中，我们读取了数据集，我们在这一步将Instances进行输入编码，每一个词都对应这一个唯一的编号，  
这些词再构成一个向量，最终形成一个(batch_size, Field)的格式喂入Model，  
我们再进行对其词向量嵌入，其变成(batch_size, num_tokens, embedding_dim)形式，  
然后使用Seq2Vec-Encoding模块，转化其为(batch_size, encoding_dim)，  
通过得到标签分类概率，最后Softmax输出进行文本分类。  
``` python
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
```

