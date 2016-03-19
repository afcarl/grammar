import data

import logging
import pickle
import os
import random

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

OUT_FILE = 'processed_data.pkl'
BOOK_FILE = 'pg2701.txt'
PARAGRAPH_LIMIT = 1000
SENT_LEN_LIMIT = 150  # chars
MIN_WORD_COUNT = 25
DIFFERENTIATE_CLOSING_PARENS = False
TEST_PROP = 0.15

def run():
    if not os.path.exists(BOOK_FILE):
        logger.info('downloading book file')
        data.get_book()

    logger.info('extracing grammers from sentences')
    sentences, grammars = data.extract_sentence_grammars(
            BOOK_FILE, num_paragraphs=PARAGRAPH_LIMIT,
            sent_len_limit=SENT_LEN_LIMIT)

    logger.info('tokenizing sentences and grammars')
    sentences = [data.tokenize_sentence(s) for s in sentences]
    grammars = [data.tokenize_grammar(g, DIFFERENTIATE_CLOSING_PARENS)
                for g in grammars]

    word2idx, words = data.create_word2idx(sentences, MIN_WORD_COUNT)
    gram2idx, grams = data.create_word2idx(grammars, min_word_count=0)

    sentences = [[word2idx.get(ss, data.UNK_ID) for ss in s]
                 for s in sentences]
    grammars = [[gram2idx[gg] for gg in g] for g in grammars]

    logger.info('splitting into train and test')
    all_data = zip(sentences, grammars)
    random.shuffle(all_data)
    split = int(len(all_data) * TEST_PROP)
    test_data = all_data[:split]
    train_data = all_data[split:]

    with open(OUT_FILE, 'wb') as out:
        pickle.dump({'train_data': train_data,
                     'test_data': test_data,
                     'words': words,
                     'word2idx': word2idx,
                     'grams': grams,
                     'gram2idx': gram2idx},
                    out)

    logger.info('train data size: %d' % len(train_data))
    logger.info('test data size: %d' % len(test_data))
    logger.info('num words: %d' % len(words))
    logger.info('num grams: %d' % len(grams))


if __name__ == '__main__':
    run()
