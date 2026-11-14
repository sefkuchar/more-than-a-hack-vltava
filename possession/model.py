"""Small team-symmetric classifier for H/A/D evidence."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier


def swap_teams(x, names):
    x=x.copy()
    for j,name in enumerate(names):
        if name.endswith('_H'):
            k=names.index(name[:-2]+'_A')
            x[:,[j,k]]=x[:,[k,j]]
        elif name=='distance_margin':
            x[:,j]*=-1
    return x


def fit_model(x,y,names,kind='hist',sample_weight=None):
    if set(np.unique(y))!={0,1,2}:
        raise ValueError('Training requires H, A and D examples')
    mirrored=swap_teams(x,names)
    train_x=np.concatenate([x,mirrored])
    train_y=np.concatenate([y,np.where(y<2,1-y,y)])
    if kind=='hist':
        model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,
            max_depth=3,learning_rate=.07,l2_regularization=10,
            min_samples_leaf=35,random_state=0)
    elif kind=='extra':
        model=ExtraTreesClassifier(n_estimators=200,max_depth=10,
            min_samples_leaf=15,max_features=.8,n_jobs=4,random_state=0)
    else:
        raise ValueError('Unknown model: '+kind)
    weights=None if sample_weight is None else np.tile(sample_weight,2)
    model.fit(train_x,train_y,sample_weight=weights)
    return model


def predict(model,x,names):
    p=model.predict_proba(x)
    q=model.predict_proba(swap_teams(x,names))[:,[1,0,2]]
    return (p+q)/2
