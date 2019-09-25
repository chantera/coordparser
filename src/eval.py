from collections import defaultdict
from enum import Enum

import chainer
import numpy as np

import teras


class Record(object):
    """
    the precision equals how many of the conjuncts output
    by the algorithm are correct, and the recall is the
    percentage of conjuncts found by the algorithm.
    [Shimbo et al, 2007]
    """

    def __init__(self):
        self.tp_t = 0
        self.tp_f = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @property
    def accuracy(self):
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp_t + self.tn) / total if total > 0 else np.nan

    @property
    def precision(self):
        denom = self.tp + self.fp
        return self.tp_t / denom if denom > 0 else np.nan

    @property
    def recall(self):
        denom = self.tp + self.fn
        return self.tp_t / denom if denom > 0 else np.nan

    @property
    def f1_score(self):
        precision = self.precision
        if precision is not np.nan:
            recall = self.recall
            if recall is not np.nan:
                denom = precision + recall
                if denom > 0:
                    return (2 * precision * recall) / denom
        return np.nan

    def __str__(self):
        return "P: {:.8f}, R: {:.8f}, F: {:.8f}" \
            .format(self.precision, self.recall, self.f1_score)

    def __repr__(self):
        return "Record(TP=({},t:{},f:{}), FP={}, FN={}, TN={})" \
            .format(self.tp, self.tp_t, self.tp_f, self.fp, self.fn, self.tn)


class Counter(object):

    class Criteria(Enum):
        WHOLE = 0
        OUTER = 1
        INNER = 2
        EXACT = 3

    OVERALL = "OVERALL"

    def __init__(self, criteria):
        assert isinstance(criteria, Counter.Criteria)
        self._criteria = criteria
        self._records = defaultdict(Record)

    def reset(self):
        self._records.clear()

    def append(self, pred_coords, true_coords):
        for cc in sorted(true_coords.keys()):
            pred_coord = pred_coords.get(cc, None)
            true_coord = true_coords[cc]
            if pred_coord is not None and true_coord is not None:
                pred_conjuncts = pred_coord.conjuncts
                true_conjuncts = true_coord.conjuncts
                coord_label = true_coord.label
                if self._criteria == Counter.Criteria.WHOLE:
                    correct = pred_conjuncts[0][0] == true_conjuncts[0][0] \
                        and pred_conjuncts[-1][1] == true_conjuncts[-1][1]
                elif self._criteria == Counter.Criteria.OUTER:
                    correct = pred_conjuncts[0] == true_conjuncts[0] \
                        and pred_conjuncts[-1] == true_conjuncts[-1]
                elif self._criteria == Counter.Criteria.INNER:
                    pred_pair = pred_coord.get_pair(cc, check=True)
                    true_pair = true_coord.get_pair(cc, check=True)
                    correct = pred_pair == true_pair
                elif self._criteria == Counter.Criteria.EXACT:
                    correct = pred_conjuncts == true_conjuncts
                self._records[Counter.OVERALL].tp += 1
                self._records[coord_label].tp += 1
                if correct:
                    self._records[Counter.OVERALL].tp_t += 1
                    self._records[coord_label].tp_t += 1
                else:
                    self._records[Counter.OVERALL].tp_f += 1
                    self._records[coord_label].tp_f += 1
            if pred_coord is not None and true_coord is None:
                self._records[Counter.OVERALL].fp += 1
            if pred_coord is None and true_coord is not None:
                coord_label = true_coord.label
                self._records[Counter.OVERALL].fn += 1
                self._records[coord_label].fn += 1
            if pred_coord is None and true_coord is None:
                self._records[Counter.OVERALL].tn += 1

    @property
    def overall(self):
        return self._records[Counter.OVERALL]


class Evaluator(teras.training.event.Listener):
    name = "evaluator"

    def __init__(self, parser, logger, report_details=False, **kwargs):
        super().__init__(**kwargs)
        self._parser = parser
        self._logger = logger
        self.report_details = report_details
        self._counter_whole = Counter(Counter.Criteria.WHOLE)
        self._counter_outer = Counter(Counter.Criteria.OUTER)
        self._counter_inner = Counter(Counter.Criteria.INNER)
        self._counter_exact = Counter(Counter.Criteria.EXACT)
        self.n_complete = 0
        self.n_sentence = 0

    def add(self, pred_coords, true_coords):
        self._counter_whole.append(pred_coords, true_coords)
        self._counter_outer.append(pred_coords, true_coords)
        self._counter_inner.append(pred_coords, true_coords)
        self._counter_exact.append(pred_coords, true_coords)
        positive_pred_coords = \
            [coord for coord in pred_coords.values() if coord is not None]
        positive_true_coords = \
            [coord for coord in true_coords.values() if coord is not None]
        if len(positive_true_coords) == 0:
            return
        if len(positive_pred_coords) == len(positive_true_coords):
            positive_pred_coords.sort(key=lambda coord: coord.cc)
            positive_true_coords.sort(key=lambda coord: coord.cc)
            if all(pred_coord == true_coord for pred_coord, true_coord
                   in zip(positive_pred_coords, positive_true_coords)):
                self.n_complete += 1
        self.n_sentence += 1

    def reset(self):
        self._counter_whole.reset()
        self._counter_outer.reset()
        self._counter_inner.reset()
        self._counter_exact.reset()
        self.n_complete = 0
        self.n_sentence = 0

    def report(self):
        counters = [("whole", self._counter_whole),
                    ("outer", self._counter_outer),
                    ("inner", self._counter_inner),
                    ("exact", self._counter_exact)]
        for name, counter in counters:
            self._logger.info("[evaluation (counter: {})] {}"
                              .format(name, counter.overall))
            self._logger.info("confusion matrix: {}"
                              .format(repr(counter.overall)))
            if self.report_details:
                details = []
                for label, record in counter._records.items():
                    if label == counter.OVERALL:
                        continue
                    details.append("\t{}: {} - {}"
                                   .format(label, str(record), repr(record)))
                self._logger.info("details:\n{}".format("\n".join(details)))
        rate = (self.n_complete / self.n_sentence) \
            if self.n_sentence > 0 else np.nan
        self._logger.info("complete: {}/{}={:.8f}"
                          .format(self.n_complete, self.n_sentence, rate))

    def on_batch_end(self, data):
        if data['train']:
            return
        true_coords = data['ts']
        with chainer.no_backprop_mode():
            pred_coords = self._parser.parse(
                *data['xs'], n_best=1, use_cache=True)
        for p_coords_i_entries, t_coords_i in zip(pred_coords, true_coords):
            if len(t_coords_i) == 0:
                assert len(p_coords_i_entries) == 0
                continue
            p_coords_i, _score = p_coords_i_entries[0]
            self.add(p_coords_i, t_coords_i)

    def on_epoch_validate_begin(self, data):
        self.reset()

    def on_epoch_validate_end(self, data):
        self.report()

    def get_overall_score(self, metric='exact'):
        if metric == 'whole':
            counter = self._counter_whole
        elif metric == 'outer':
            counter = self._counter_outer
        elif metric == 'inner':
            counter = self._counter_inner
        elif metric == 'exact':
            counter = self._counter_exact
        else:
            raise ValueError('invalid metric: {}'.format(metric))
        return counter.overall.f1_score
