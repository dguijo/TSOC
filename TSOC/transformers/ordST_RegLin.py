""" shapelet transformers
transformer from the time domain into the shapelet domain. Standard full transform, a contracted version and
a randoms sampler
"""
__author__ = ["Jason Lines", "David Guijo"]
__all__=["OrdinalShapeletTransformRegLin","ContractedOrdinalShapeletTransformRegLin","RandomEnumerationOrdinalShapeletTransformRegLin","Shapelet","ShapeletPQ"]

import os
import time
import warnings
import numpy as np
import pandas as pd

from itertools import zip_longest
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import heapq
from operator import itemgetter
from sktime.transformers.base import BaseTransformer
from scipy.stats import spearmanr

from scipy.stats import linregress
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)


class OrdinalShapeletTransformRegLin(BaseTransformer):

    """Shapelet Transform.

    Original journal publication:
    @article{hills2014classification,
      title={Classification of time series by shapelet transformation},
      author={Hills, Jon and Lines, Jason and Baranauskas, Edgaras and Mapp, James and Bagnall, Anthony},
      journal={Data Mining and Knowledge Discovery},
      volume={28},
      number={4},
      pages={851--881},
      year={2014},
      publisher={Springer}
    }

    Parameters
    ----------
    min_shapelet_length                 : int, lower bound on candidatie shapelet lengths (default = 3)
    max_shapelet_length                 : int, upper bound on candidatie shapelet lengths (default = inf or series length)
    max_shapelets_to_store_per_class    : int, upper bound on number of shapelets to retain from each distinct class (default = 200)
    random_state                        : RandomState, int, or none: to control reandom state objects for deterministic results (default = None)
    verbose                             : int, level of output printed to the console (for information only) (default = 0)
    remove_self_similar                 : boolean, remove overlapping "self-similar" shapelets from the final transform (default = True)

    Attributes
    ----------

    predefined_ig_rejection_level       : float, minimum information gain required to keep a shapelet (default = 0.05)
    self.shapelets                      : list of Shapelet objects, the stored shapelets after a dataest has been processed
    """

    def __init__(self,
             min_shapelet_length=3,
             max_shapelet_length=np.inf,
             max_shapelets_to_store_per_class=200,
             random_state=None,
             verbose=0,
             remove_self_similar=True,
             l_norm = 1
             ):

        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.max_shapelets_to_store_per_class = max_shapelets_to_store_per_class
        self.random_state = random_state
        self.verbose = verbose
        self.remove_self_similar = remove_self_similar
        self.predefined_ig_rejection_level = 0.0
        self.shapelets = None
        self.l_norm = l_norm

    def fit(self, X, y=None):
        """A method to fit the shapelet transform to a specified X and y

        Parameters
        ----------
        X: pandas DataFrame
            The training input samples.
        y: array-like or list
            The class values for X

        Returns
        -------
        self : FullShapeletTransform
            This estimator
        """
        if self.time_limit <= 0:
            raise ValueError("Error: time limit cannot be equal to or less than 0")

        X_lens = np.array([len(X.iloc[r,0]) for r in range(len(X))]) # note, assumes all dimensions of a case are the same length. A shapelet would not be well defined if indices do not match!
        X = np.array([[X.iloc[r,c].values for c in range(len(X.columns))] for r in range(len(X))]) # may need to pad with nans here for uneq length, look at later

        num_ins = len(y)
        distinct_class_vals = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        candidates_evaluated = 0

        if type(self) is RandomEnumerationOrdinalShapeletTransformRegLin:
            num_series_to_visit = min(self.num_cases_to_sample, len(y))
        else:
            num_series_to_visit = num_ins

        shapelet_heaps_by_class = {i: ShapeletPQ() for i in distinct_class_vals}

        self.random_state = check_random_state(self.random_state)

        # Here we establish the order of cases to sample. We need to sample x cases and y shapelets from each (where x = num_cases_to_sample
        # and y = num_shapelets_to_sample_per_case). We could simply sample x cases without replacement and y shapelets from each case, but
        # the idea is that if we are using a time contract we may extract all y shapelets from each x candidate and still have time remaining.
        # Therefore, if we get a list of the indices of the series and shuffle them appropriately, we can go through the list again and extract
        # another y shapelets from each series (if we have time).

        # We also want to ensure that we visit all classes so we will visit in round-robin order. Therefore, the code below extracts the indices
        # of all series by class, shuffles the indices for each class independently, and then combines them in alternating order. This results in
        # a shuffled list of indices that are in alternating class order (e.g. 1,2,3,1,2,3,1,2,3,1...)

        def _round_robin(*iterables):
            sentinel = object()
            return (a for x in zip_longest(*iterables, fillvalue=sentinel) for a in x if a != sentinel)

        case_ids_by_class = {i: np.where(y == i)[0] for i in distinct_class_vals}

        # if transform is random/contract then shuffle the data initially when determining which cases to visit
        if type(self) is RandomEnumerationOrdinalShapeletTransformRegLin or type(self) is ContractedOrdinalShapeletTransformRegLin:
            for i in range(len(distinct_class_vals)):
                self.random_state.shuffle(case_ids_by_class[distinct_class_vals[i]])

        num_train_per_class = {i : len(case_ids_by_class[i]) for i in case_ids_by_class}
        round_robin_case_order = _round_robin(*[list(v) for k, v in case_ids_by_class.items()])
        cases_to_visit = [(i, y[i]) for i in round_robin_case_order]
        # this dictionary will be used to store all possible starting positions and shapelet lengths for a give series length. This
        # is because we enumerate all possible candidates and sample without replacement when assessing a series. If we have two series
        # of the same length then they will obviously have the same valid shapelet starting positions and lengths (especially in standard
        # datasets where all series are equal length) so it makes sense to store the possible candidates and reuse, rather than
        # recalculating each time

        # Initially the dictionary will be empty, and each time a new series length is seen the dict will be updated. Next time that length
        # is used the dict will have an entry so can simply reuse
        possible_candidates_per_series_length = {}

        # a flag to indicate if extraction should stop (contract has ended)
        time_finished = False

        # max time calculating a shapelet
        # for timing the extraction when contracting
        start_time = time.time()
        time_taken = lambda: time.time() - start_time
        max_time_calc_shapelet = -1
        time_last_shapelet = time_taken()

        # for every series
        case_idx = 0
        while case_idx < len(cases_to_visit):

            series_id = cases_to_visit[case_idx][0]
            this_class_val = cases_to_visit[case_idx][1]

            # minus 1 to remove this candidate from sums
            binary_ig_this_class_count = num_train_per_class[this_class_val]-1
            binary_ig_other_class_count = num_ins-binary_ig_this_class_count-1

            if self.verbose:
                if type(self) == RandomEnumerationOrdinalShapeletTransformRegLin:
                    print("visiting series: " + str(series_id) + " (#" + str(case_idx + 1) + "/" + str(num_series_to_visit) + ")")
                else:
                    print("visiting series: " + str(series_id) + " (#" + str(case_idx + 1) + ")")

            this_series_len = len(X[series_id][0])

            # The bound on possible shapelet lengths will differ series-to-series if using unequal length data.
            # However, shapelets cannot be longer than the series, so set to the minimum of the series length
            # and max shapelet length (which is inf by default)
            if self.max_shapelet_length == -1:
                this_shapelet_length_upper_bound = this_series_len
            else:
                this_shapelet_length_upper_bound = min(this_series_len, self.max_shapelet_length)

            # all possible start and lengths for shapelets within this series (calculates if series length is new, a simple look-up if not)
            # enumerate all possible candidate starting positions and lengths.

            # First, try to reuse if they have been calculated for a series of the same length before.
            candidate_starts_and_lens = possible_candidates_per_series_length.get(this_series_len)
            # else calculate them for this series length and store for possible use again
            if candidate_starts_and_lens is None:
                candidate_starts_and_lens = [
                    [start, length] for start in range(0, this_series_len - self.min_shapelet_length + 1)
                    for length in range(self.min_shapelet_length, this_shapelet_length_upper_bound + 1) if start + length <= this_series_len]
                possible_candidates_per_series_length[this_series_len] = candidate_starts_and_lens

            # default for full transform
            candidates_to_visit = candidate_starts_and_lens
            num_candidates_per_case = len(candidate_starts_and_lens)

            # limit search otherwise:
            if hasattr(self,"num_candidates_to_sample_per_case"):
                num_candidates_per_case = min(self.num_candidates_to_sample_per_case, num_candidates_per_case)
                cand_idx = list(self.random_state.choice(list(range(0, len(candidate_starts_and_lens))), num_candidates_per_case, replace=False))
                candidates_to_visit = [candidate_starts_and_lens[x] for x in cand_idx]

            for candidate_idx in range(num_candidates_per_case):

                # if shapelet heap for this class is not full yet, set entry criteria to be the predetermined IG threshold
                ig_cutoff = self.predefined_ig_rejection_level
                # otherwise if we have max shapelets already, set the threshold as the IG of the current 'worst' shapelet we have
                if shapelet_heaps_by_class[this_class_val].get_size() >= self.max_shapelets_to_store_per_class:
                    ig_cutoff = max(shapelet_heaps_by_class[this_class_val].peek()[0], ig_cutoff)

                cand_start_pos = candidates_to_visit[candidate_idx][0]
                cand_len = candidates_to_visit[candidate_idx][1]

                candidate = OrdinalShapeletTransformRegLin.zscore(X[series_id][:,cand_start_pos: cand_start_pos + cand_len])

                # now go through all other series and get a distance from the candidate to each
                orderline = []

                # initialise here as copy, decrease the new val each time we evaluate a comparison series
                num_visited_this_class = 0
                num_visited_other_class = 0

                candidate_rejected = False

                for comparison_series_idx in range(len(cases_to_visit)):
                    i = cases_to_visit[comparison_series_idx][0]

                    if y[i] != cases_to_visit[comparison_series_idx][1]:
                        raise ValueError("class match sanity test broken")

                    bsf_dist = np.inf

                    start_left = cand_start_pos
                    start_right = cand_start_pos+1

                    if X_lens[i]==cand_len:
                        start_left = 0
                        start_right = 0

                    for num_cals in range(max(1,int(np.ceil((X_lens[i]-cand_len)/2)))): # max used to force iteration where series len == candidate len
                        if start_left < 0:
                            start_left = X_lens[i]-1-cand_len

                        comparison = OrdinalShapeletTransformRegLin.zscore(X[i][:,start_left: start_left+ cand_len])
                        dist_left = np.linalg.norm(candidate-comparison)
                        bsf_dist = min(dist_left*dist_left, bsf_dist)

                        # for odd lengths
                        if start_left == start_right:
                            continue

                        # right
                        if start_right == X_lens[i]-cand_len+1:
                            start_right = 0
                        comparison = OrdinalShapeletTransformRegLin.zscore(X[i][:,start_right: start_right + cand_len])
                        dist_right = np.linalg.norm(candidate-comparison)
                        bsf_dist = min(dist_right*dist_right, bsf_dist)

                        start_left-=1
                        start_right+=1

                    orderline.append((bsf_dist, y[i]))

                    """
                    if len(orderline) > 2:
                        corr_upper_bound = OrdinalShapeletTransform.calc_early_correlation(orderline, y[series_id], cases_to_visit)
                        if corr_upper_bound <= ig_cutoff:
                            candidate_rejected = True
                            break
                    """
                candidates_evaluated += 1
                if self.verbose > 3 and candidates_evaluated % 100 == 0:
                    print("candidates evaluated: " + str(candidates_evaluated))

                # only do if candidate was not rejected
                if candidate_rejected is False:
                    final_ig = OrdinalShapeletTransformRegLin.calc_correlation_regLin(orderline, y[series_id], self.l_norm)
                    accepted_candidate = Shapelet(series_id, cand_start_pos, cand_len, final_ig, candidate)

                    # add to min heap to store shapelets for this class
                    shapelet_heaps_by_class[this_class_val].push(accepted_candidate)

                    # informal, but extra 10% allowance for self similar later
                    if shapelet_heaps_by_class[this_class_val].get_size() > self.max_shapelets_to_store_per_class*3:
                        shapelet_heaps_by_class[this_class_val].pop()

                # Takes into account the use of the MAX shapelet calculation time to not exceed the time_limit (not exact, but likely a good guess).
                if hasattr(self,'time_limit') and self.time_limit > 0:
                    time_now = time_taken()
                    time_this_shapelet = (time_now - time_last_shapelet)
                    if time_this_shapelet > max_time_calc_shapelet:
                        max_time_calc_shapelet = time_this_shapelet
                    time_last_shapelet = time_now
                    if (time_now + max_time_calc_shapelet) > self.time_limit:
                        if self.verbose > 0:
                            print("No more time available! It's been {0:02d}:{1:02}".format(int(round(time_now / 60, 3)), int((round(time_now / 60, 3) - int(round(time_now / 60, 3))) * 60)))
                        time_finished = True
                        break
                    else:
                        if self.verbose > 0:
                            if candidate_rejected is False:
                                print("Candidate finished. {0:02d}:{1:02} remaining".format(int(round((self.time_limit - time_now) / 60, 3)),
                                                                                            int((round((self.time_limit - time_now) / 60, 3) - int(round((self.time_limit - time_now) / 60, 3))) * 60)))
                            else:
                                print("Candidate rejected. {0:02d}:{1:02} remaining".format(int(round((self.time_limit - time_now) / 60, 3)),
                                                                                            int((round((self.time_limit - time_now) / 60, 3) - int(round((self.time_limit - time_now) / 60, 3))) * 60)))

            # stopping condition: in case of iterative transform (i.e. num_cases_to_sample have been visited)
            #                     in case of contracted transform (i.e. time limit has been reached)
            case_idx += 1

            if case_idx >= num_series_to_visit:
                if hasattr(self,'time_limit') and time_finished is not True:
                    case_idx = 0
            elif case_idx >= num_series_to_visit or time_finished:
                if self.verbose > 0:
                    print("Stopping search")
                break

        # remove self similar here
        # for each class value
        #       get list of shapelets
        #       sort by quality
        #       remove self similar

        self.shapelets = []
        for class_val in distinct_class_vals:
            by_class_descending_ig = sorted(shapelet_heaps_by_class[class_val].get_array(), key=itemgetter(0), reverse=True)

            if self.remove_self_similar and len(by_class_descending_ig) > 0:
                by_class_descending_ig = OrdinalShapeletTransformRegLin.remove_self_similar_shapelets(by_class_descending_ig)
            else:
                # need to extract shapelets from tuples
                by_class_descending_ig = [x[2] for x in by_class_descending_ig]

            # if we have more than max_shapelet_per_class, trim to that amount here
            if len(by_class_descending_ig) > self.max_shapelets_to_store_per_class:
                by_class_descending_ig = by_class_descending_ig[:self.max_shapelets_to_store_per_class]

            self.shapelets.extend(by_class_descending_ig)

        # final sort so that all shapelets from all classes are in descending order of information gain
        self.shapelets.sort(key=lambda x:x.info_gain, reverse=True)

    @staticmethod
    def remove_self_similar_shapelets(shapelet_list):
        """Remove self-similar shapelets from an input list. Note: this method assumes
        that shapelets are pre-sorted in descending order of quality (i.e. if two candidates
        are self-similar, the one with the later index will be removed)

        Parameters
        ----------
        shapelet_list: list of Shapelet objects

        Returns
        -------
        shapelet_list: list of Shapelet objects
        """

        # IMPORTANT: it is assumed that shapelets are already in descending order of quality. This is preferable in the fit method as removing self-similar
        # shapelets may be False so the sort needs to happen there in those cases, and avoids a second redundant sort here if it is set to True

        def is_self_similar(shapelet_one, shapelet_two):
            # not self similar if from different series
            if shapelet_one.series_id != shapelet_two.series_id:
                return False
            if (shapelet_one.start_pos >= shapelet_two.start_pos) and (shapelet_one.start_pos <= shapelet_two.start_pos + shapelet_two.length):
                return True
            if (shapelet_two.start_pos >= shapelet_one.start_pos) and (shapelet_two.start_pos <= shapelet_one.start_pos + shapelet_one.length):
                return True

        # [s][2] will be a tuple with (info_gain,id,Shapelet), so we need to access [2]
        to_return = [shapelet_list[0][2]]  # first shapelet must be ok

        for s in range(1, len(shapelet_list)):
            can_add = True
            for c in range(0, s):
                if is_self_similar(shapelet_list[s][2], shapelet_list[c][2]):
                    can_add = False
                    break
            if can_add:
                to_return.append(shapelet_list[s][2])

        return to_return

    # transform a set of data into distances to each extracted shapelet
    def transform(self, X, **transform_params):
        """Transforms X according to the extracted shapelets (self.shapelets)

        Parameters
        ----------
        X : pandas DataFrame
            The input dataframe to transform

        Returns
        -------
        output : pandas DataFrame
            The transformed dataframe in tabular format.
        """
        if self.shapelets is None:
            raise Exception("Fit not called yet or no shapelets were generated")

        X = np.array([[X.iloc[r, c].values for c in range(len(X.columns))] for r in range(len(X))])  # may need to pad with nans here for uneq length, look at later

        output = np.zeros([len(X), len(self.shapelets)], dtype=np.float32, )

        # for the i^th series to transform
        for i in range(0, len(X)):
            this_series = X[i]

            # get the s^th shapelet
            for s in range(0, len(self.shapelets)):
                # find distance between this series and each shapelet
                min_dist = np.inf
                this_shapelet_length = self.shapelets[s].length

                for start_pos in range(0, len(this_series[0]) - this_shapelet_length + 1):
                    comparison = OrdinalShapeletTransformRegLin.zscore(this_series[:, start_pos:start_pos + this_shapelet_length])

                    dist = np.linalg.norm(self.shapelets[s].data - comparison)
                    dist = dist*dist
                    dist = 1.0/this_shapelet_length*dist
                    min_dist = min(min_dist, dist)

                    output[i][s] = min_dist

        return pd.DataFrame(output)

    def fit_transform(self, X, y=None, **fit_params):
        """Fits and transforms a given input X and y

        Parameters
        ----------
        X: pandas.DataFrame the input data to transform
        y: list or array like of class values corresponding to the indices in X

        Returns
        -------
        Xt : pandas DataFrame
            The transformed pandas DataFrame.
        """
        if self.shapelets is None and y is not None:
            self.fit(X, y)
        elif self.shapelets is not None:
            raise Exception("Trying to fit but shapelets already exist.")
        else:
            raise Exception("No class values specified - shapelet extraction is supervised and requires Y to build the transform")

        return self.transform(X)

    def get_shapelets(self):
        """An accessor method to return the extracted shapelets

        Returns
        -------
        shapelets: a list of Shapelet objects
        """
        return self.shapelets

    @staticmethod
    def binary_entropy(num_this_class, num_other_class):
            ent = 0
            if num_this_class != 0:
                ent -= num_this_class / (num_this_class + num_other_class) * np.log2(num_this_class / (num_this_class + num_other_class))
            if num_other_class != 0:
                ent -= num_other_class / (num_this_class + num_other_class) * np.log2(num_other_class / (num_this_class + num_other_class))
            return ent

    @staticmethod
    def calc_fisher_nominal(orderline):

        orderline = np.array(orderline)

        media_general = np.mean(orderline[:, 0])

        numerador = 0
        denominador = 0
        labels, counts = np.unique(orderline[:, 1], return_counts=True)

        for i, j in enumerate(labels):
            numerador += (counts[i] * np.power(np.mean(orderline[orderline[:, 1] == j, 0]) - media_general, 2))
            denominador += (counts[i] * np.std(orderline[orderline[:, 1] == j, 0]))

        return numerador/denominador


    @staticmethod
    def calc_fisher_ordinal(orderline):

        orderline = np.array(orderline)

        numerador = 0
        denominador = 0
        labels, counts = np.unique(orderline[:, 1], return_counts=True)

        for k in labels:
            denominador += np.std(orderline[orderline[:, 1] == k, 0])
            media_distancias_k = np.mean(orderline[orderline[:, 1] == k, 0])
            for j in labels:
                media_distancias_j = np.mean(orderline[orderline[:, 1] == j, 0])
                numerador += (np.abs(k - j) * np.power(media_distancias_k - media_distancias_j, 2))

        return numerador/((len(labels) - 1) * denominador)

    @staticmethod
    def calc_correlation(orderline, shp_class):

        orderline = np.array(orderline)
        r2_value = r2_score(orderline[:, 0], np.abs(orderline[:, 1] - shp_class))

        return r2_value

    @staticmethod
    def calc_correlation_regLin(orderline, shp_class, l_norm):

        orderline = np.array(orderline)

        _, _, r_value, _, _ = linregress(orderline[:, 0], np.power(np.abs(orderline[:, 1] - shp_class), l_norm))
        return r_value ** 2

    """
    @staticmethod
    def calc_early_correlation(orderline, shp_class, y):

        orderline = np.array(orderline)

        y = list(dict(y).values()) #extract classes in the correct order of round robin

        if len(orderline) < len(y):
            orderline_aux = np.vstack((orderline, np.zeros((len(y) - len(orderline), 2))))

            for i in range(len(orderline), len(y)):
                orderline_aux[i, :] = [np.abs(y[i] - shp_class), y[i]]

        else:
            orderline_aux = orderline
        r2_value = r2_score(orderline_aux[:, 0], np.abs(orderline_aux[:, 1] - shp_class))

        return r2_value
    """
    # could cythonise
    @staticmethod
    def calc_binary_ig(orderline, total_num_this_class, total_num_other_class):
        # def entropy(ent_class_counts, all_class_count):

        initial_ent = OrdinalShapeletTransformRegLin.binary_entropy(total_num_this_class, total_num_other_class)
        bsf_ig = 0

        count_this_class = 0
        count_other_class = 0

        total_all = total_num_this_class+total_num_other_class

        # evaluate each split point
        for split in range(0, len(orderline) - 1):
            next_class = orderline[split][1] # +1 if this class, -1 if other
            if next_class > 0:
                count_this_class += 1
            else:
                count_other_class += 1

            # optimistically add this class to left side first and other to right
            left_prop = (split + 1) / total_all
            ent_left = OrdinalShapeletTransformRegLin.binary_entropy(count_this_class,count_other_class)

            right_prop = 1-left_prop # because right side must optimistically contain everything else
            ent_right = OrdinalShapeletTransformRegLin.binary_entropy(total_num_this_class-count_this_class,total_num_other_class-count_other_class)

            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

        return bsf_ig

    # could cythonise
    @staticmethod
    def calc_early_binary_ig(orderline, num_this_class_in_orderline, num_other_class_in_orderline, num_to_add_this_class, num_to_add_other_class):
        # def entropy(ent_class_counts, all_class_count):

        initial_ent = OrdinalShapeletTransformRegLin.binary_entropy(num_this_class_in_orderline+num_to_add_this_class, num_other_class_in_orderline+num_to_add_other_class)
        bsf_ig = 0

        # actual observations in orderline
        count_this_class = 0
        count_other_class = 0

        total_all = num_this_class_in_orderline+num_other_class_in_orderline+num_to_add_this_class+num_to_add_other_class

        # evaluate each split point
        for split in range(0, len(orderline) - 1):
            next_class = orderline[split][1] # +1 if this class, -1 if other
            if next_class > 0:
                count_this_class += 1
            else:
                count_other_class += 1

            # optimistically add this class to left side first and other to right
            left_prop = (split + 1 + num_to_add_this_class) / total_all
            ent_left = OrdinalShapeletTransformRegLin.binary_entropy(count_this_class+num_to_add_this_class,count_other_class)

            right_prop = 1-left_prop # because right side must optimistically contain everything else
            ent_right = OrdinalShapeletTransformRegLin.binary_entropy(num_this_class_in_orderline-count_this_class,num_other_class_in_orderline-count_other_class+num_to_add_other_class)

            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

            # now optimistically add this class to right, other to left
            left_prop = (split + 1 + num_to_add_other_class) / total_all
            ent_left = OrdinalShapeletTransformRegLin.binary_entropy(count_this_class,count_other_class+num_to_add_other_class)

            right_prop = 1-left_prop # because right side must optimistically contain everything else
            ent_right = OrdinalShapeletTransformRegLin.binary_entropy(num_this_class_in_orderline-count_this_class+num_to_add_this_class,num_other_class_in_orderline-count_other_class)
            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

        return bsf_ig

    @staticmethod
    def zscore(a, axis=0, ddof=0):
        """ A static method to return the normalised version of series.  This mirrors the scipy implementation
        with a small difference - rather than allowing /0, the function returns output = np.zeroes(len(input)).
        This is to allow for sensible processing of candidate shapelets/comparison subseries that are a straight
        line. Original version: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html

        Parameters
        ----------
        a : array_like
            An array like object containing the sample data.

        axis : int or None, optional
            Axis along which to operate. Default is 0. If None, compute over the whole array a.

        ddof : int, optional
            Degrees of freedom correction in the calculation of the standard deviation. Default is 0.

        Returns
        -------
        zscore : array_like
            The z-scores, standardized by mean and standard deviation of input array a.
        """
        zscored = np.empty(a.shape)
        for i, j in enumerate(a):
            # j = np.asanyarray(j)
            sstd = j.std(axis=axis, ddof=ddof)

            # special case - if shapelet is a straight line (i.e. no variance), zscore ver should be np.zeros(len(a))
            if sstd == 0:
                zscored[i] = np.zeros(len(j))
            else:
                mns = j.mean(axis=axis)
                if axis and mns.ndim < j.ndim:
                    zscored[i] = ((j - np.expand_dims(mns, axis=axis)) /
                                    np.expand_dims(sstd, axis=axis))
                else:
                    zscored[i] = (j - mns) / sstd
        return zscored

    @staticmethod
    def euclidean_distance_early_abandon(u, v, min_dist):
        sum_dist = 0
        for i in range(0, len(u[0])):
            for j in range(0, len(u)):
                u_v = u[j][i] - v[j][i]
                sum_dist += np.dot(u_v, u_v)
                if sum_dist >= min_dist:
                    # The distance is higher, so early abandon.
                    return min_dist
        return sum_dist


class ContractedOrdinalShapeletTransformRegLin(OrdinalShapeletTransformRegLin):
    __author__ = "Jason Lines and David Guijo"

    """Contracted Shapelet Transform.
    @incollection{bostrom2017binary,
      title={Binary shapelet transform for multiclass time series classification},
      author={Bostrom, Aaron and Bagnall, Anthony},
      booktitle={Transactions on Large-Scale Data-and Knowledge-Centered Systems XXXII},
      pages={24--46},
      year={2017},
      publisher={Springer}
    }

    Parameters
    ----------
    min_shapelet_length                 : int, lower bound on candidatie shapelet lengths (default = 3)
    max_shapelet_length                 : int, upper bound on candidatie shapelet lengths (default = inf or series length)
    max_shapelets_to_store_per_class    : int, upper bound on number of shapelets to retain from each distinct class (default = 200)
    time_limit_in_mins                  : float, the number of minutes allowed for shapelet extraction (default = 60)
    num_candidates_to_sample_per_case   : int, number of candidate shapelets to assess per training series before moving on to 
                                          the next series (default = 20)
    random_state                        : RandomState, int, or none: to control reandom state objects for deterministic results (default = None)
    verbose                             : int, level of output printed to the console (for information only) (default = 0)
    remove_self_similar                 : boolean, remove overlapping "self-similar" shapelets from the final transform (default = True)

    Attributes
    ----------

    predefined_ig_rejection_level       : float, minimum information gain required to keep a shapelet (default = 0.05)
    self.shapelets                      : list of Shapelet objects, the stored shapelets after a dataest has been processed
    """

    def __init__(
            self,
            min_shapelet_length = 3,
            max_shapelet_length = np.inf,
            max_shapelets_to_store_per_class = 200,
            time_limit_in_mins=60,
            num_candidates_to_sample_per_case = 20,
            random_state = None,
            verbose = 0,
            remove_self_similar = True,
            l_norm = 1
    ):

        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.max_shapelets_to_store_per_class = max_shapelets_to_store_per_class
        self.num_candidates_to_sample_per_case = num_candidates_to_sample_per_case
        self.time_limit = time_limit_in_mins*60
        self.random_state = random_state
        self.verbose = verbose
        self.remove_self_similar = remove_self_similar
        self.predefined_ig_rejection_level = 0.0
        self.shapelets = None
        self.l_norm = l_norm


class RandomEnumerationOrdinalShapeletTransformRegLin(OrdinalShapeletTransformRegLin):
    pass
    # to follow


class Shapelet:
    """A simple class to model a Shapelet with associated information

    Parameters
    ----------
    series_id: int
        The index of the series within the data (X) that was passed to fit.
    start_pos: int
        The starting position of the shapelet within the original series
    length: int
        The length of the shapelet
    info_gain: flaot
        The calculated information gain of this shapelet
    data: array-like
        The (z-normalised) data of this shapelet
    """

    def __init__(self, series_id, start_pos, length, info_gain, data):
        self.series_id = series_id
        self.start_pos = start_pos
        self.length = length
        self.info_gain = info_gain
        self.data = data

    def __str__(self):
        return "Series ID: {0}, start_pos: {1}, length: {2}, info_gain: {3}, ".format(self.series_id, self.start_pos, self.length, self.info_gain)


class ShapeletPQ:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, shapelet):
        heapq.heappush(self._queue, (shapelet.info_gain, self._index, shapelet))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def peek(self):
        return self._queue[0]

    def get_size(self):
        return len(self._queue)

    def get_array(self):
        return self._queue


def write_transformed_data_to_arff(transform, labels, file_name):
    """ A simple function to save the transform obtained in arff format

    Parameters
    ----------
    transform: array-like
        The transform obtained for a dataset
    labels: array-like
        The labels of the dataset
    file_name: string
        The directory to save the transform
    """
    # Create directory in case it doesn't exists
    directory = '/'.join(file_name.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_shapelets = transform.shape[1]
    unique_labels = np.unique(labels).tolist()

    with open(file_name, 'w+') as f:
        # Headers
        f.write("@Relation Shapelets" + file_name.split('/')[-1].split('_')[0] + '\n\n')
        for i in range(0, num_shapelets):
            f.write("@attribute Shapelet_" + str(i) + " numeric\n")
        f.write("@attribute target {" + ",".join(unique_labels) + "}\n")
        f.write("\n@data\n")
        # Patterns
        for i, j in enumerate(transform):
            pattern = j.tolist() + [int(float(labels[i]))]
            f.write(",".join(map(str, pattern)) + "\n")
    f.close()


def write_shapelets_to_csv(shapelets, data, dim_to_use, time, file_name):
    """ A simple function to save the shapelets obtained in csv format

    Parameters
    ----------
    shapelets: array-like
        The shapelets obtained for a dataset
    data: array-like
        The original data
    time: fload
        The time spent obtaining shapelets
    file_name: string
        The directory to save the set of shapelets
    """
    data = data.iloc[:, dim_to_use]

    data_aux = [[]] * len(data)
    for i in range(0, len(data)):
        data_aux[i] = np.array([np.asarray(x) for x in np.asarray(data.iloc[i, :])])
    data = data_aux.copy()

    # Create directory in case it doesn't exists
    directory = '/'.join(file_name.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, 'w+') as f:
        # Number of shapelets and time extracting
        f.write(str(len(shapelets)) + "," + str(time) + "\n")
        for i, j in enumerate(shapelets):
            f.write(str(j.info_gain) + "," + str(j.series_id) + "," + ''.join(str(j.dims)).replace(', ', ':') + "," + str(j.start_pos) + "," + str(j.length) + "\n")
            for k in range(0, len(dim_to_use)):
                f.write(",".join(map(str, data[j.series_id][k, j.start_pos:j.start_pos + j.length])) + "\n")
    f.close()
