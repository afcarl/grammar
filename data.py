import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import subprocess

PARSER_DIR = '~/berkeleyparser'


def get_book():
    """Downloads 'Moby Dick'"""
    subprocess.check_ouput(
        ['wget', 'http://www.gutenberg.org/cache/epub/2701/pg2701.txt'])


def read_paragraphs(book_file, limit=1000):
    """Reads file and returns list of paragraphs.

    Assumes paragraphs are delimited by blank lines.

    """
    paragraphs = []
    current_str = ''
    with open(book_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            # check for end of paragraph
            if '' == stripped and current_str.strip():
                paragraphs.append(current_str.strip())
                current_str = ''
            if stripped:
                current_str += stripped + ' '
            if len(paragraphs) >= limit:
                break
        else:
            # add last paragraph
            if current_str:
                paragraphs.append(current_str.strip())
    return paragraphs


def compute_grammar(sentences):
    """Uses Berkeley Parser to compute grammar trees on list of sentences."""
    if not sentences:
        return []
    my_parser = os.path.join(PARSER_DIR, 'BerkeleyParser-1.7.jar')
    my_grammar = os.path.join(PARSER_DIR, 'eng_sm6.gr')
    p = subprocess.Popen(
        ' '.join(['java', '-jar', my_parser, '-gr', my_grammar]),
        stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=None,
        shell=True)

    stdout = p.communicate(input='\n'.join(sentences))[0]
    lines = stdout.split('\n')
    if len(lines) != len(sentences):  # last line is empty leftover
        assert len(lines) == 1 + len(sentences)
        lines = lines[:-1]
    return lines


def extract_sentence_grammars(book_file, num_paragraphs=1000, sent_len_limit=150):
    """Extracts sentences and their grammars from book file."""
    paragraphs = read_paragraphs(book_file, limit=num_paragraphs)
    # attempt to find first paragraph of book
    for first_idx, paragraph in enumerate(paragraphs):
        if 'Call me Ishmael.' in paragraph:
            break
    else:
        first_idx = 0
    paragraphs = paragraphs[first_idx:]

    # tokenize sentences
    sentences = []
    for paragraph in paragraphs:
        para_sentences = sent_tokenize(paragraph)
        sentences.extend([sent for sent in para_sentences
                          if sent.strip() and len(sent) <= sent_len_limit
                          and '(' not in sent and ')' not in sent])

    grammars = compute_grammar(sentences)
    return sentences, grammars


def tokenize_sentence(s, lower=True):
    """Tokenize a sentence into words."""
    if lower:
        s = s.lower()
    return word_tokenize(s)


def tokenize_grammar(g, differentiate_closing_parens=True):
    """Tokenize linear parse tree g."""
    g = g.replace(')', ' )')
    init_tokens = g.split()
    out_tokens = []
    path = []
    for tok in init_tokens:
        if ')' in tok:
            assert tok == ')'
            assert path
            if differentiate_closing_parens:
                complement = path.pop()
                out_tokens.append(complement.replace('(', ')'))
            else:
                out_tokens.append(')')
        elif '(' in tok:
            path.append(tok)
            out_tokens.append(tok)
        else:  # leaf node is just a word
            out_tokens.append('XX')
    return out_tokens


def create_word2idx(tokenized_sentences, min_word_count=25):
    """Given list of tokenized sentences, returns proper indices for embeddings."""
    word_counts = {}
    for tokens in tokenized_sentences:
        for tok in tokens:
            if tok not in word_counts:
                word_counts[tok] = 0
            word_counts[tok] += 1

    words = []
    word2idx = {}
    for word, count in word_counts.iteritems():
        if count < min_word_count:
            continue
        word2idx[word] = len(words)
        words.append(word)

    return word2idx, words
