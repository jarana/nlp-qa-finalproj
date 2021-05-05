"""Prints QA examples.

Author:
    Shrey Desai and Yasumasa Onoe
"""

from collections import Counter
import argparse
import random
import textwrap
import stanza
from stanza.server import CoreNLPClient
from nltk.tree import *


from termcolor import colored

from data import QADataset


RULE_LENGTH = 100
DOC_WIDTH = 100
TEXT_WRAPPER = textwrap.TextWrapper(width=DOC_WIDTH)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--path',
    type=str,
    default='datasets/squad_dev.jsonl.gz',
    required=False,
    help='path to display samples from',
)
parser.add_argument(
    '--samples',
    type=int,
    default=10,
    required=False,
    help='number of samples to visualize',
)
parser.add_argument(
    '--shuffle',
    action='store_true',
    help='whether to shuffle samples before displaying',
)
parser.add_argument(
    '--max_context_length',
    type=int,
    default=384,
    help='maximum context length (do not change!)',
)
parser.add_argument(
    '--max_question_length',
    type=int,
    default=64,
    help='maximum question length (do not change!)',
)


def _build_string(tokens):
    """Builds string from token list."""

    return ' '.join(tokens)


def _color_context(context, answer_start, answer_end):
    """Colors answer span with bold + underline red within the context."""

    tokens = []

    i = 0
    while i < len(context):
        if i == answer_start:
            span = _build_string(context[answer_start:(answer_end + 1)])
            tokens.append(
                colored(span, 'red', attrs=['bold', 'underline']),
            )
            i = answer_end + 1
        else:
            tokens.append(context[i])
            i += 1

    lines = TEXT_WRAPPER.wrap(text=' '.join(tokens))

    return '\n'.join(lines)

def _generate_txs(annots):
    transitions = []
    
    for sent in annots.sentences[:2]:
        print(sent.parseTree)
        stack = []
        dumb = []
        left = True
        cts = Counter([w.head for w in sent.words])# help ensure to only start reducing once last child reached
        for word in sent.words:
            stack.append(word.head)
            dumb.append(0)# always shift first ? TODO double check
            print(f'shifting {word.head} on to stack')
            if stack[-1] == 0:
                left = False
            elif len(stack) > 1 and (stack[-2] > 0):
                if left and (stack[-2] < stack[-1] or stack[-1] < word.id) or (not left and stack[-2] > stack[-1]):
                    print(f'reducing {stack[-2]} & {stack[-1]}')
                    print(f'old stack: {stack}')
                    curr = stack.pop()
                    stack[-1] = curr
                    print(f'new stack: {stack}')
                    dumb.append(1)

                while len(stack) > 1 and stack[-2] == stack[-1]:
                    print(f'reducing {stack[-2]} & {stack[-1]}')
                    print(f'old stack: {stack}')
                    stack.pop()
                    print(f'new stack: {stack}')
                    dumb.append(1)
        # at this point the stack should have left, root, right
        while len(stack) > 1:
            print(f'final reducing {stack[-2]} & {stack[-1]}')
            print(f'old stack: {stack}')
            stack.pop()
            print(f'new stack: {stack}')
            dumb.append(1)
        print([w.text for w in sent.words])
        print([w.id for w in sent.words])
        print([w.head for w in sent.words])
        ind = 0
        for w in sent.words:
            print(f'id: {w.id} - text: {w.text} - head: {w.head}')
        txs = ['SHIFT', 'REDUCE']
        print([txs[t] for t in dumb])
        transitions.extend(dumb)

def cnlp_to_nltk_tree(t):
    return Tree("", [cnlp_to_nltk_tree(c) for c in t.child]) if t.child else t.value

def prepare_tree(t):
    tree = cnlp_to_nltk_tree(t)
    tree.pprint()
    #re.sub("\s\s+", " ", raw)
    return " ".join(str(tree).replace(")", " )").split()).replace("(","").split()

def generate_transitions(flat_tree):
    txs = []
    last = ""
    for item in flat_tree:
        if item == ")" and last != ")":
            txs.append(1)
        elif item != ")":
            txs.append(0)
        last = item
    return txs

def main(args):
    """Visualization of contexts, questions, and colored answer spans."""

    # Load dataset, and optionally shuffle.
    dataset = QADataset(args, args.path)
    samples = dataset.samples
    if args.shuffle:
        random.shuffle(samples)

    vis_samples = samples[:args.samples]

    print()
    print('-' * RULE_LENGTH)
    print()

    # Visualize samples.
    for (qid, context, question, answer_start, answer_end) in vis_samples:
        cxt = _build_string(context)
        print(cxt)
        #stanza.download('en')
        #en_nlp = stanza.Pipeline('en')
        #en_doc = en_nlp(cxt)
        with CoreNLPClient(annotators=['parse'], timeout=30000, memory='16G') as client:
            ann = client.annotate(cxt)
        print([tok.word for tok in ann.sentence[0].token])
        tree = prepare_tree(ann.sentence[0].parseTree)
        print(tree)
        txs = generate_transitions(tree)
        print(txs)

        

if __name__ == '__main__':
    main(parser.parse_args())
