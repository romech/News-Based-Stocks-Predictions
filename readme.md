## Work in progress

### Loading Stanford CoreNLP
```bash
wget https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
```

Text tokenizing example ([halp](https://nlp.stanford.edu/software/tokenizer.html))

```bash
java -cp "./stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.process.PTBTokenizer -preserveLines -lowerCase -filter "[a-zA-Z]|[^\w]+|(\d{4}\-\d\d\-\d\d,\d)" -options "ptb3Escaping=false" < data/intermediate/test_data_0.csv > output.txt
```