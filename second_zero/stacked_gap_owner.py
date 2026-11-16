"""Team-symmetric raw motion, observed/uncertain geometry and crossfit evidence."""
import numpy as np
from second_zero.global_team_context import features as motion
from second_zero.temporal_ownership import trajectory_features
from second_zero.sprint_model_interpolation import interpolated_features

MODEL=dict(max_iter=240,max_leaf_nodes=15,max_depth=5,min_samples_leaf=80,learning_rate=.05,l2_regularization=10.,random_state=20261030,early_stopping=False)
CHANNELS=425


def features(raw,canonical):
    if set(raw)!={'ball','players','team','pitch'}:raise ValueError('Only four raw fields allowed')
    n=len(raw['ball']);x=np.asarray(canonical,np.float32)
    if x.shape!=(n,13):raise ValueError('Canonical evidence requires13 columns')
    q=np.repeat(x[:,None],2,axis=1);q[:,0,[0,1,3,7,8,9]]*=-1
    result=np.concatenate([motion(raw['players'],raw['team'],raw['pitch']),trajectory_features(raw['ball'],raw['players'],raw['team'],raw['pitch']),interpolated_features(raw['ball'],raw['players'],raw['team'],raw['pitch']),q],axis=2).astype(np.float32)
    if result.shape!=(n,2,CHANNELS) or not np.isfinite(result).all():raise ValueError('Invalid stacked owner feature matrix')
    return result
