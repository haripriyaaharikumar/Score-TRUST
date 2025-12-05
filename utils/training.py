from six import moves, iteritems
from os.path import exists
import shutil
import heapq as hq
from operator import itemgetter
import numpy as np
import os

class ContinuousIndexSampler(object):
    """
    V1.0: Stable running
    V1.1: Add seed and local RandomState
    """
    def __init__(self, data_size_or_ids, sample_size, shuffle=False, seed=None):
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.seed = seed
        self._rs = np.random.RandomState(self.seed)

        if isinstance(data_size_or_ids, int):
            self.ids = list(range(data_size_or_ids))
        else:
            assert hasattr(data_size_or_ids, '__len__')
            self.ids = list(data_size_or_ids)

        self.sids = []
        while len(self.sids) < self.sample_size:
            self.sids += self._renew_ids()
        self.pointer = 0

    def _renew_ids(self):
        rids = [idx for idx in self.ids]
        if self.shuffle:
            self._rs.shuffle(rids)
        return rids

    def sample_ids(self):
        if self.pointer + self.sample_size > len(self.sids):
            self.sids = self.sids[self.pointer:] + self._renew_ids()
            while len(self.sids) < self.sample_size:
                self.sids += self._renew_ids()
            self.pointer = 0

        return_ids = self.sids[self.pointer: self.pointer + self.sample_size]
        self.pointer += self.sample_size
        return return_ids

    def sample_ids_continuous(self):
        while True:
            if self.pointer + self.sample_size > len(self.sids):
                self.sids = self.sids[self.pointer:] + self._renew_ids()
                while len(self.sids) < self.sample_size:
                    self.sids += self._renew_ids()
                self.pointer = 0

            return_ids = self.sids[self.pointer: self.pointer + self.sample_size]
            self.pointer += self.sample_size
            yield return_ids

def get_oldest_save_dir(parent_dir, name_prefix):
    largest_step = -1
    oldest_save_dir = None

    path_names = os.listdir(parent_dir)
    for path_name in path_names:
        full_path = os.path.join(parent_dir, path_name)
        if os.path.isdir(full_path) and path_name.startswith(name_prefix):
            step = int(path_name[len(name_prefix):])
            if step > largest_step:
                largest_step = step
                oldest_save_dir = full_path
    return oldest_save_dir

class BestResultsTracker:
    def __init__(self, keys_and_cmpr_types, num_best):
        """
        keys: A list of strings
        num_best: Number of best results we want to keep track
        """

        avail_cmpr_types = ("less", "greater")
        assert isinstance(keys_and_cmpr_types, (tuple, list)), \
            "'keys_and_cmpr_types' must be a list of 2-lists/tuples of the form (key, cmpr_type)!"
        for n in moves.xrange(len(keys_and_cmpr_types)):
            assert isinstance(keys_and_cmpr_types[n], (tuple, list)) and len(keys_and_cmpr_types[n]) == 2, \
                "The element {} of 'keys_and_cmpr_types' must be a list/tuple of length 2!".format(n)
            assert isinstance(keys_and_cmpr_types[n][0], (str, bytes)), \
                "The element {} of 'keys_and_cmpr_types' must have 'key' to be str or bytes!".format(n)
            assert keys_and_cmpr_types[n][1] in avail_cmpr_types, \
                "The element {} of 'keys_and_cmpr_types' must have 'type' in {}!".format(n, avail_cmpr_types)

        self.keys_and_cmpr_type = keys_and_cmpr_types
        self.num_best = num_best

        self._best_results = dict()
        for key, cmpr_type in keys_and_cmpr_types:
            inits = [(-1e10, -1e10, -1) for _ in range(self.num_best)]
            hq.heapify(inits)

            cmpr_type = 1.0 if cmpr_type == "greater" else -1.0
            self._best_results[key] = (cmpr_type, inits)

    def check_and_update(self, results, step, assert_keys=False):
        assert isinstance(results, dict), \
            f"'results' must be a dict. Found {type(results)}!"

        is_better = dict()
        print (results)
        for key, val in iteritems(results):
            stored_results = self._best_results.get(key)

            if stored_results is None:
                if assert_keys:
                    raise ValueError(f"The values of '{key}' are not tracked!")
                else:
                    is_better[key] = None
            else:
                cmpr_type, item_heap = stored_results
                new_item = (cmpr_type * val, val, step)

                old_item = hq.heappushpop(item_heap, new_item)

                if old_item[-1] == step:
                    is_better[key] = False
                else:
                    is_better[key] = True

        return is_better

    def check_and_update_key(self, key, val, step):
        stored_results = self._best_results.get(key)

        if stored_results is None:
            raise ValueError(f"The values of '{key}' are not tracked!")
        else:
            cmpr_type, item_heap = stored_results
            new_item = (cmpr_type * val, val, step)

            old_item = hq.heappushpop(item_heap, new_item)

            if old_item[-1] == step:
                is_better = False
            else:
                is_better = True

        return is_better

    def get_best_results(self, sort_results=False):
        results = dict()

        for key, (cmpr_type, item_heap) in iteritems(self._best_results):
            if sort_results:
                item_heap = sorted(item_heap, key=itemgetter(0), reverse=True)
            results[key] = [item[1:] for item in item_heap]

        return results

    def set_best_results(self, results):
        self._validate_results(results)
        self._best_results = results

    def _validate_results(self, results):
        assert isinstance(results, dict), f"'results' must be a dict. Found {type(results)}!"
        for key, cmpr_type in self.keys_and_cmpr_type:
            val = results.get(key)

            assert val is not None, "'results' do not contain key {}!".format(key)
            assert isinstance(val, tuple), \
                f"'results[{key}]' is not a tuple. " \
                f"Found type(results[{key}])={type(val)}!"
            assert len(val) == 2, f"len(results[{key}])={val}"

            if cmpr_type == "greater":
                assert val[0] == 1.0, f"'cmpr_type' for key '{key}' is " \
                    f"'greater' but val[0] in 'results[{key}]' is {val[0]}!"
            else:
                assert val[0] == -1.0, f"'cmpr_type' for key '{key}' is " \
                    f"'less' but val[0] in 'results[{key}]' is {val[0]}!"


class SaveDirTracker:
    def __init__(self, max_save, dir_path_prefix):
        self.max_save = max_save
        self._saved_steps = []
        self._dir_path_prefix = dir_path_prefix

    def get_save_dir(self, step):
        return self._dir_path_prefix + "%d" % step

    def update_and_delete_old_save(self, step):
        self._saved_steps.append(step)
        if len(self._saved_steps) > self.max_save:
            old_step = self._saved_steps.pop(0)
            old_dir = self.get_save_dir(old_step)
            if exists(old_dir):
                shutil.rmtree(old_dir)

    def get_saved_steps(self):
        return self._saved_steps

    def set_saved_steps(self, steps):
        self._validate_steps(steps)
        self._saved_steps = steps

    def _validate_steps(self, steps):
        assert isinstance(steps, list), "'steps' must be a list. Found {}!".format(type(steps))
        assert len(steps) <= self.max_save, \
            "'len(steps)' must be <= {}. Found {}!".format(self.max_save, len(steps))

        for i, step in enumerate(steps):
            assert isinstance(step, int), "'steps[{}]' must be an int. Found {}!".format(i, type(step))
