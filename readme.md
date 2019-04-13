## Work in progress

### Loading Stanford CoreNLP
```bash
wget https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
```

Text tokenizing example

```bash
java -cp "./stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.process.PTBTokenizer -preserveLines -lowerCase -filter "(^\w){1,2}|(\d{4}\-\d\d\-\d\d,\d)" < input.txt > output.txt
```