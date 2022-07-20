from lfqa_utils import *
from nlp import load_dataset
from datasets import load_dataset
import datasets
from datasets import load_metric
from datasets import set_caching_enabled
from rake_nltk import Rake
from transformers.generation_logits_process import LogitsWarper
from transformers import LogitsProcessorList,TopKLogitsWarper,TopPLogitsWarper
from transformers import TemperatureLogitsWarper

set_caching_enabled(True)
eli5 = datasets.load_dataset('eli5',cache_dir = './.cache/huggingface/datasets/')
wiki40b_snippets = load_dataset('wiki_snippets',name='wiki40b_en_100_0',cache_dir = './.cache/huggingface/datasets/')['train']
# print("found")
# assert False
# training arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ArgumentsQAR():
    def __init__(self):
        self.batch_size = 512
        self.max_length = 128
        self.checkpoint_batch_size = 32
        self.print_freq = 100
        self.pretrained_model_name = "google/bert_uncased_L-8_H-768_A-12"
        self.model_save_name = "retriever_models/eli5_retriever_model_l-8_h-768_b-512-512"
        self.learning_rate = 2e-4
        self.num_epochs = 10

qar_args = ArgumentsQAR()

# prepare torch Dataset objects
qar_train_dset = ELI5DatasetQARetriver(eli5['train_eli5'], training=True)
qar_valid_dset = ELI5DatasetQARetriver(eli5['validation_eli5'], training=False)
print("found")
# load pre-trained BERT and make model
qar_tokenizer, qar_model = make_qa_retriever_model(
        model_name=qar_args.pretrained_model_name,
        from_file=None,
        device=device
)
###
from transformers import RagRetriever
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path.faiss,
)
#####
# train the model
# train_qa_retriever(qar_model, qar_tokenizer, qar_train_dset, qar_valid_dset, qar_args)

qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to(device)
_ = qar_model.eval()
print("Evaluating")

if not os.path.isfile('wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat'):
    make_qa_dense_index(
        qar_model, qar_tokenizer, wiki40b_snippets, device=device,
        index_name='wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat')
faiss_res = faiss.StandardGpuResources()
wiki40b_passage_reps = np.memmap(
            'wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat',
            dtype='float32', mode='r',
            shape=(wiki40b_snippets.num_rows, 128)
)

wiki40b_index_flat = faiss.IndexFlatIP(128)
wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
wiki40b_gpu_index.add(wiki40b_passage_reps)

question = eli5['test_eli5'][12345]['title']
doc, res_list = query_qa_dense_index(question, qar_model, qar_tokenizer, wiki40b_snippets, wiki40b_gpu_index, device=device)

df = pd.DataFrame({
    'Article': ['---'] + [res['article_title'] for res in res_list],
    'Sections': ['---'] + [res['section_title'] if res['section_title'].strip() != '' else res['article_title']
                 for res in res_list],
    'Text': ['--- ' + question] + [res['passage_text'] for res in res_list],
})
# print(df.head)



# pre-computing support documents

qa_s2s_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')
_ = qa_s2s_model.eval()


def keyword_tokens(text,tokenizer):
    r = Rake()
    r.extract_keywords_from_text(text)
    q_toks = tokenizer.batch_encode_plus(r.get_ranked_phrases(), max_length=10)
    word_tokens = []
    for i in q_toks['input_ids']:
        i.pop(0)
        l = len(i)
        i.pop(l-1)
        if i not in word_tokens:
            word_tokens.append(i)
    # print(word_tokens)
    single_tokens = set()
    for i in word_tokens:
        for j in i:
            single_tokens.add(j)
    # print(single_tokens)
    first_tokens = set()
    for i in word_tokens:
        first_tokens.add(i[0])
    # print(first_tokens)
    return word_tokens,list(single_tokens),list(first_tokens)


class KeyWordWraper(LogitsWarper):
    def __init__(self,Lambda: float, single_tokens:List[int] ,word_tokens:List[List[int]],key_continue=10):
        self.single_tokens = single_tokens
        self.word_tokens = word_tokens
        self.Lambda = Lambda
        self.key_continue = key_continue
    def equal(self, prev_tokens: List[int], word_token: List[int]):
        if len(word_token) == 0:
            return False
        elif len(word_token) > len(prev_tokens): 
            return False
        else:
            return prev_tokens[-len(word_token) :] == word_token
    def _important_tokens(self, prev_input_ids: List[int],important_token_seq:List[int]):
        important_tokens = []
        for index,i in enumerate(important_token_seq):
            if self.equal(prev_input_ids[-1-index:],important_token_seq[:index+1])\
                      and index >= 2:
                return important_token_seq[index]
        return None
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for word_token in self.word_tokens:
            if len(word_token) >= 3 and self._important_tokens(input_ids[0].tolist(),word_token) != None:
                print("found big fish")
                scores[:, self._important_tokens(input_ids[0].tolist(),word_token)] *= self.key_continue
        keyword_mask = torch.zeros(scores.shape[1])
        # print(input_ids)
        for i in self.single_tokens:
            if i not in input_ids[0].tolist():
                scores[:, i] *= self.Lambda
        return scores


questions = []
answers = []
golds = []
import random
# print(eli5['test_eli5'].num_rows)
print("TopPLogitsWarper(0.95)+keys")
random.seed(420)
# print(random.choice(eli5['test_eli5'].num_rows))
# assert False
for j in range(100):
# for i in [12345] + [j for j in range(4)]:
    i = random.choice(range(eli5['test_eli5'].num_rows))
    # create support document with the dense index
    # print("question :",j)
    question = eli5['test_eli5'][i]['title']
    # question = "who is yann lecun?"
    gold = eli5['test_eli5'][i]['answers']['text'][0]
    score = eli5['test_eli5'][i]['answers']['score'][0];
    doc, res_list = query_qa_dense_index(
        question, qar_model, qar_tokenizer,
        wiki40b_snippets, wiki40b_gpu_index, device=device
    )
    # concatenate question and support document into BART input
    question_doc = "question: {} context: {}".format(question, doc)
    word_tokens,single_tokens,first_tokens = keyword_tokens(question_doc,qa_s2s_tokenizer)
    # assert False
    # generate an answer with beam search
    logits_warper = LogitsProcessorList([
    # TopKLogitsWarper(30),
    # TopKLogitsWarper(40),
    # TemperatureLogitsWarper(0.75),
    # TemperatureLogitsWarper(0.5),
    # TopPLogitsWarper(0.9),
    TopPLogitsWarper(0.95),
    KeyWordWraper(2,first_tokens,word_tokens)
    ])
    answer,out = qa_s2s_generate(
            question_doc, qa_s2s_model, qa_s2s_tokenizer,
            do_sample = True,
            # top_k=40,
            num_answers=1,
            # num_beams=8,
            min_len=64,
            max_len = 256,
            #max_new_tokens=256,
            max_input_length=1024,
            logits_processor = logits_warper,
            device=device
    )
    # print("A :"+  answer[0])
    questions += [question]
    answers += [answer[0]]
    golds += [gold]
    
df = pd.DataFrame({
    'Question': questions,
    'Answer': answers,
})
# df.style.set_properties(**{'text-align': 'left'})
# print(df.head)
# assert False
rouge = datasets.load_metric('rouge')
meteor = datasets.load_metric('meteor')
results = rouge.compute(predictions = answers, references=golds,)
print(list(results.keys()))
print(results)

m_results = meteor.compute(predictions = answers, references=golds)
print(round(m_results['meteor'], 2))
