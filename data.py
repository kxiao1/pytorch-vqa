import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import config
import utils
import colors


def get_loader(train=False, val=False, test=False, color_only=False, check_suitable=False,
    check_pertains_to_color=False, include_original_images=False):
    """ Returns a data loader for the desired split """
    assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'

    split = VQA(
        utils.path_for_annotations(train=train, val=val, test=test),
        config.test_preprocessed_path if test else config.preprocessed_path,
        config.test_unprocessed_path if test else config.unprocessed_images_path,
        answerable_only=train,
        color_only=color_only,
        check_suitable=check_suitable,
        check_pertains_to_color=check_pertains_to_color,
        include_original_images=include_original_images,
    )
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class VQA(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, annotations_path, image_features_path, image_path, answerable_only=False,
            color_only=False, check_suitable = False, check_pertains_to_color=False, include_original_images=False):
        super(VQA, self).__init__()
        with open(annotations_path, 'r') as fd:
            annotations_json = json.load(fd)
        with open(config.vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)

        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']

        # q and a
        self.questions = list(prepare_questions(annotations_json))
        self.answers = list(prepare_answers(annotations_json))
        assert len(self.questions) == len(self.answers)
        self.questions = [self._encode_question(q) for q in self.questions]
        
        self.check_suitable = check_suitable
        if self.check_suitable:
            # only check if question is unanswerable 
            self.answers = [self._encode_suitable(a) for a in self.answers]
            print(len(self.answers), len(self.answers)-sum(self.answers))
        else:
            self.answers = [self._encode_answers(a) for a in self.answers]
        

        # v
        self.image_features_path = image_features_path
        self.image_path = image_path
        self.include_original_images = include_original_images
        self.vizwiz_id_to_index = self._create_vizwiz_id_to_index()
        self.vizwiz_ids = [int(i["image"][-12:-4]) for i in annotations_json]

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable()

        # self.color_only: only use questions that pertain to color?
        # self.check_pertains_to_color: include in the output
        # a binary flag indicating whether the question has at least one
        # answer which is a color
        self.color_only = color_only
        self.check_pertains_to_color = check_pertains_to_color
        if self.color_only or self.check_pertains_to_color:
            import colors
            self.color_answer_indices = [self.answer_to_index[color] for color in colors.colors]
            self.color_question_indices = self._find_color_question_indices()
            self.color_question_indices_set = set(self.color_question_indices)

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions))
        return self._max_length

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def _create_vizwiz_id_to_index(self):
        """ Create a mapping from a VizWiz image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            vizwiz_ids = features_file['ids'][()]
        vizwiz_id_to_index = {id: i for i, id in enumerate(vizwiz_ids)}
        return vizwiz_id_to_index

    def _find_answerable(self):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        for i, answers in enumerate(self.answers):
            answer_has_index = len(answers.nonzero()) > 0
            # store the indices of anything that is answerable
            if answer_has_index:
                answerable.append(i)
        return answerable

    def _find_color_question_indices(self):
        """ Create a list of indices into questions that have at least one answer that's a color """
        color_indices = []
        for i, answers in enumerate(self.answers):
            answer_has_color = sum(answers[self.color_answer_indices]) > 0
            # store the indices of anything that has a color as an answer
            if answer_has_color:
                color_indices.append(i)
        return color_indices

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
        return vec, len(question)

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.answer_to_index))
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

    def _encode_suitable(self, answers):
        """ Turn an answer into a binary flag: 0 iff at least 3 out 10 answers were 'unsuitable'"""
        num_bad = 0
        for answer in answers:
            if answer == "unsuitable" or answer == "unsuitable image":
                num_bad += 1
        return torch.tensor(0) if num_bad >= 3 else torch.tensor(1)

    def _load_image_features(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.vizwiz_id_to_index[image_id]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')
        return torch.from_numpy(img)

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'images_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.images_file = h5py.File(self.image_path, 'r')
        index = self.vizwiz_id_to_index[image_id]
        dataset = self.images_file['features']
        img = dataset[index].astype('float32')
        return torch.from_numpy(img)

    def __getitem__(self, item):
        if self.answerable_only:
            # change of indices to only address answerable questions
            item = self.answerable[item]
        if self.color_only:
            # change of indices to only address questions about color
            item = self.color_question_indices[item]

        q, q_length = self.questions[item]
        a = self.answers[item]
        if self.color_only:
            a = a[self.color_answer_indices]
        image_id = self.vizwiz_ids[item]
        if self.include_original_images:
            v = (self._load_image_features(image_id), self._load_image(image_id))
        else:
            v = self._load_image_features(image_id)
        # since batches are re-ordered for PackedSequence's, the original question order is lost
        # we return `item` so that the order of (v, q, a) triples can be restored if desired
        # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.

        if self.check_pertains_to_color:
            c = 1 if item in self.color_question_indices_set else 0
            return v, q, c, item, q_length

        return v, q, a, item, q_length

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        elif self.color_only:
            return len(self.color_question_indices)
        else:
            return len(self.questions)


# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

def process_punctuation(s):
    # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
    # this version should be faster since we use re instead of repeated operations on str's
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()

def prepare_questions(annotations_json):
    """ Tokenize and normalize questions from a given annotations json in the usual VQA format. """
    questions = [image['question'] for image in annotations_json]
    for question in questions:
        question = process_punctuation(question.lower())
        yield question.strip().split()


def prepare_answers(annotations_json):
    """ Normalize answers from a given annotations json in the usual VQA format. """
    if 'answers' not in annotations_json[0]:
        for _ in range(len(annotations_json)):
            yield [""]*10
        return
    answers = [[a['answer'] for a in image['answers']] for image in annotations_json]
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96
    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))


class VizWizImages(data.Dataset):
    """ Dataset for VizWiz images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(VizWizImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        # Format: VizWiz_{train/test}_{id}.jpg
        # parse id as an integer
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class Composite(data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))
