import preprocess
import seq2seq


def run():
    seq2seq.decode(preprocess.OUT_FILE)

if __name__ == '__main__':
    run()
