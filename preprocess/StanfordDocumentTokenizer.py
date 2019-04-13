import os
import tempfile
from utils.config import Config

config = Config.open("preproc.config.yml")


def tokenize(string):
    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path = os.path.join(tmpdir, 'test.in')
        out_path = os.path.join(tmpdir, 'test.out')
        with open(inp_path, "w") as inp:
            inp.write(string)
        tokenize_file(inp_path, out_path)
        with open(out_path, "r") as out:
            for line in out:
                print(line)


def tokenize_file(input_path, output_path):
    cmd = build_shell_command(input_path, output_path)
    print('Executing', cmd)
    return_code = os.system(cmd)
    if return_code != 0:
        print('Stanford Tokenizer returned error code', return_code)


def build_shell_command(input_path, output_path):
    return (r'java -cp "{}" edu.stanford.nlp.process.PTBTokenizer ' +
            r'{} -filter "{}" -options "{}" < "{}" > "{}"').format(
        config("stanford-tokenizer.classpath"),
        ' '.join(f'-{param}' for param in config("stanford-tokenizer.parameters")),
        '|'.join(config("stanford-tokenizer.exclude")),
        ','.join(config("stanford-tokenizer.options")),
        input_path,
        output_path)


def run():
    tokenize_file(config("source.combined"), config("output_folder"))

if __name__ == '__main__':
    tokenize("U.S. gosdolg is $100'000'000'000'000'000, how's that for you, @ElonMusk?")
