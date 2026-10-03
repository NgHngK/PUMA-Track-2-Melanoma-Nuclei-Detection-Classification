from __future__ import annotations
import numpy as np
# Reconstructed local-V17 contract from the P1-P7 master report. This file is not claimed byte-identical
# to the unavailable historical source. Strict <15 px, same-class matching, GT-order, score/distance/stable-order.

def match_one_roi(gt,pred,radius=15.0,n_classes=10):
    gt=list(gt); pred=[dict(x,_stable=i) for i,x in enumerate(pred)]; alive=list(range(len(pred))); tp=np.zeros(n_classes,int)
    for g in gt:
        c=int(g['class_id']); cand=[]
        for ai in alive:
            p=pred[ai]
            if int(p['class_id'])!=c: continue
            d=float(np.hypot(float(p['x'])-float(g['x']),float(p['y'])-float(g['y'])))
            if d<radius: cand.append((-float(p.get('score',1.0)),d,int(p['_stable']),ai))
        if cand:
            _,_,_,chosen=min(cand); tp[c]+=1
            # Historical deletion searched first remaining prediction with same centroid, not necessarily UID.
            cp=pred[chosen]; delete=None
            for ai in alive:
                if int(pred[ai]['class_id'])==c and float(pred[ai]['x'])==float(cp['x']) and float(pred[ai]['y'])==float(cp['y']): delete=ai; break
            if delete is not None: alive.remove(delete)
    gt_count=np.bincount([int(x['class_id']) for x in gt],minlength=n_classes); pred_count=np.bincount([int(x['class_id']) for x in pred],minlength=n_classes)
    return {'tp':tp,'fp':pred_count-tp,'fn':gt_count-tp}
def _f1(tp,fp,fn):
    den=2*tp+fp+fn; return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
def evaluate_rois(gt_by_roi,pred_by_roi,roi_order=None,n_classes=10):
    rois=list(roi_order or sorted(set(gt_by_roi)|set(pred_by_roi))); mats=[]
    for r in rois: mats.append(match_one_roi(gt_by_roi.get(r,[]),pred_by_roi.get(r,[]),n_classes=n_classes))
    roi_f1=np.stack([_f1(x['tp'],x['fp'],x['fn']) for x in mats]); tp=sum((x['tp'] for x in mats),np.zeros(n_classes,int)); fp=sum((x['fp'] for x in mats),np.zeros(n_classes,int)); fn=sum((x['fn'] for x in mats),np.zeros(n_classes,int))
    return {'roi_fixed10_macro_f1':float(roi_f1.mean(1).mean()) if len(rois) else 0.0,'pooled_fixed10_macro_f1':float(_f1(tp,fp,fn).mean()),'tp':tp.tolist(),'fp':fp.tolist(),'fn':fn.tolist(),'roi_order':rois}
