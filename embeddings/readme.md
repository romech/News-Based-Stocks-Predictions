# FastText
[Getting Started with FastText](https://fasttext.cc/docs/en/support.html)
### Building
```bash
wget https://github.com/facebookresearch/fastText/archive/v0.2.0.zip
unzip v0.2.0.zip -q
cd fastText-0.2.0 && make
```

### View list of comands

```bash
./fastText-0.2.0/fasttext
```

### Download a pre-trained model

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

## Getting vectors for news
### Make a file with plain article
```python
import pandas as pd

df = pd.read_table("data/source/RedditNews.tsv")

df.News.to_csv("news_only.txt", header=False, index=False)
```

### Use FastText
```bash
./fastText-0.2.0/fasttext print-sentence-vectors cc.en.300.bin < news_only.txt > data/prepared/article_vectors.txt
```
