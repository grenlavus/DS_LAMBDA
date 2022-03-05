# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """
    GATING_THRESHOLD = np.sqrt(kalman_filter.chi2inv95[4]) #### Cuidado !!! ele esqueceu que está setando uma dependência 
    #com kalman filter - olhar lá: se only position = True vai dar errado

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, _lambda=0.5):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        #print(matches)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _full_cost_metric(self, tracks, dets, track_indices, detection_indices):
        """
        This implements the full lambda-based cost-metric. However, in doing so, it disregards
        the possibility to gate the position only which is provided by
        linear_assignment.gate_cost_matrix(). Instead, I gate by everything.

        Note that the Mahalanobis distance is itself an unnormalised metric. Given the cosine
        distance being normalised, we employ a quick and dirty normalisation based on the
        threshold: that is, we divide the positional-cost by the gating threshold, thus ensuring
        that the valid values range 0-1.

        Note also that the authors work with the squared distance. I also sqrt this, so that it
        is more intuitive in terms of values.
        """
        # Compute First the Position-based Cost Matrix
        print("ENTREI NO FULL_COST")
        pos_cost = np.empty([len(track_indices), len(detection_indices)]) #Return a new array of given shape and type, without initializing entries
        #track_indices=lines detection_indices=col
        msrs = np.asarray([dets[i].to_xyah() for i in detection_indices]) #Convert the input detections to an array
        #of type xyah (ele entende que o método .to_xyah está relacionado ao objeto definido por Detection)
        for row, track_idx in enumerate(track_indices):# row é a variável que guarda os índices das tracks e 
        #track_idx é á própria track (ver enumerate()) - contador e conteúdo
            pos_cost[row, :] = np.sqrt( # para cada índice, ele calcula a  
                self.kf.gating_distance(tracks[track_idx].mean, tracks[track_idx].covariance, msrs, False) # distância de mahalanobis
                ) / self.GATING_THRESHOLD
            # by divinding by self.GATING_THRESHOLD the pos_cost will be 1 at 95% Confidence Level
            
            #print("track_mean=",tracks[track_idx].mean, "track_cov=",tracks[track_idx].covariance)    
            #print("index=",track_idx,"cost_pos=",pos_cost)
            #print("GATING_Treshold="self.GATING_THRESHOLD)    
        pos_gate = pos_cost > 1.0 #return true if pos_cost >1
        print("pos_gate=",pos_gate)
        # Now Compute the Appearance-based Cost Matrix
        #x = input("hit any Number ")
        app_cost = self.metric.distance(
            np.array([dets[i].feature for i in detection_indices]),
            np.array([tracks[i].track_id for i in track_indices]),
        )
        app_gate = app_cost > self.metric.matching_threshold
        print("app_gate=",app_gate)
        # Now combine and threshold
        cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost #Aqui ele faz a distribuição de pesos com lambda
        print("cost_matrix=",cost_matrix) #printa a função custo para cada objeto detectado
        cost_matrix[np.logical_or(pos_gate, app_gate)] = linear_assignment.INFTY_COST
        # Se um dos gates for TRUE, ou seja se distância de mahalanobis for > do valor que representa 95% de nível de confiança
        # ou então se a métrica do coseno for maior que o threshold então ele "joga" o 
        # custo para INFTY_COST (1e+5) e desvincula a associação, senão ele considera que a detecção foi associada a track corretamente
                      
        return cost_matrix

    def _match(self, detections):
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()] # if TrackState=2
        #print("confirmed_tracks=",confirmed_tracks)
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()] # if TrackState=1 or 3

        # Associate confirmed tracks using appearance features -> como a função _full_cost_metric mudou, agora essa etapa deve 
        # usar kalman + appearance features
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            self._full_cost_metric,
            linear_assignment.INFTY_COST - 1,  # no need for self.metric.matching_threshold here,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection): # entra aqui se a detecção e nova e se trata de uma nova track (por exemplo primeiro frame)
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature, class_name))
        self._next_id += 1
        #print("não entra aqui a não ser na primeira rodada")
