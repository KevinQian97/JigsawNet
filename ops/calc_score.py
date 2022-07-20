from torch.nn import functional as F
import torch

def calc_er_contrast_score_cache(output,model,args,indices,atom_dic,act_labels):
    '''
    calc the prob scores for loss calc

    output {
        "vision":tensor(B*video_candidates X D)
        "cls_emb": tensor(sum(B,M) X D)  
        "obj_emb": tensor(B*video_candidates X D)
        "obj_clf_emb": list(B X tensor(20XD))
        }
    '''
    #resize vision,obj_emb and cls_emb
    # print(output["vision"].size(),output["vision"].requires_grad)
    # print(output["obj_emb"].size(),output["obj_emb"].requires_grad)
    # print(output["cls_emb"].size(),output["cls_emb"].requires_grad)
    # print(output["obj_clf_emb"].size(),output["obj_clf_emb"].requires_grad)
    # print(act_labels)
    # raise RuntimeError("size check")
    if args.video_candidates >1:
        video_candidates = args.video_candidates
    elif args.video_candidates == 0:
        video_candidates = args.num_segments
    else:
        video_candidates = 4
    batch_size = int(output["vision"].size(0)/video_candidates)
    vision = output["vision"].chunk(batch_size,0)
    objs = output["obj_emb"].chunk(batch_size,0)
    acts = output["cls_emb"].split(indices.tolist())
    obj_clfs = output["obj_clf_emb"].chunk(batch_size,0)
    act_mat = []
    act_logits = []
    obj_logits = []
    er_act_logits = []
    er_obj_logits = []
    consist_loss = 0

    for act_id,v in atom_dic.items():
        act_mat.append(model.module.act_cache[v].data)
    
    for v,obj_clf,cur_act,act_label in zip(vision,obj_clfs,acts,act_labels):
        act_logit = []
        er_act_logit = v.mm(F.normalize(obj_clf.squeeze()).T).max(0).values

        # consist_loss = consist_loss + (v.mm(v.T).sum()-v.size(0))/2

        hard_neg_scores = v.mm(model.module.act_cache.data.T).sort()
        ato_ids = atom_dic[act_label.item()].cuda()
        for i in range(v.size(0)):
            for j in range(1,model.module.act_cache.size(1)+1):
                if hard_neg_scores.indices[i][-j] not in ato_ids:
                    consist_loss = consist_loss+hard_neg_scores.values[i][-j]
                    # print(consist_loss.size())
                    break

        er_act_logits.append(er_act_logit)
        act_mat[act_label] = cur_act
        for idx in range(len(act_mat)):
            act = act_mat[idx]
            logit = v.mm(act.T).max(1)
            if idx == act_label.item():
                ato_ids = atom_dic[idx][logit.indices]
                for jdx in range(len(ato_ids)):
                    ato_id  = ato_ids[jdx]
                    if ato_id not in model.module.vis_cache:
                        model.module.vis_cache[ato_id] = {"feat":model.module.act_cache[ato_id].data.unsqueeze(0),"score":0.3}
                    if logit.values[jdx]>model.module.vis_cache[ato_id]["score"]:
                        model.module.vis_cache[ato_id] = {"feat":v[jdx].data.unsqueeze(0),"score":logit.values[jdx]}
                    consist_loss = consist_loss+1-v[jdx].unsqueeze(0).mm(model.module.vis_cache[ato_id]["feat"].T).squeeze()
            act_logit.append(logit.values.mean(0))
        act_logit = torch.stack(act_logit,0)
        act_logits.append(act_logit)
    act_logits = torch.stack(act_logits,0)
    er_act_logits = torch.stack(er_act_logits,0)

    for obj,obj_clf,cur_act,act_label in zip(objs,obj_clfs,acts,act_labels):
        obj_logit = []
        er_obj_logit = obj.mm(F.normalize(obj_clf.squeeze()).T).max(0).values
        er_obj_logits.append(er_obj_logit)
        act_mat[act_label] = cur_act
        for act in act_mat:
            obj_logit.append(obj.mm(act.T).max(1).values.mean(0))
        obj_logit = torch.stack(obj_logit,0)
        obj_logits.append(obj_logit)
    obj_logits = torch.stack(obj_logits,0)
    er_obj_logits = torch.stack(er_obj_logits,0)

    #update model.act_cache
    for cur_act,act_label in zip(acts,act_labels):
        model.module.act_cache[atom_dic[act_label.item()]] = cur_act.data

    # print(act_logits.size(),act_logits.requires_grad)
    # print(obj_logits.size(),obj_logits.requires_grad)
    # print(er_act_logits.size(),er_act_logits.requires_grad)
    # print(er_obj_logits.size(),er_obj_logits.requires_grad)
    # print(model.module.act_cache,model.module.act_cache.requires_grad)
    # raise RuntimeError("size check")
    return act_logits,obj_logits,er_act_logits,er_obj_logits,consist_loss/batch_size

def calc_score(output,model,args,indices,atom_dic):
    '''
    calc the prob scores

    output {
        "vision":tensor(B*video_candidates X D)
        "cls_emb": tensor(sum(B,M) X D)  
        "obj_emb": tensor(B*video_candidates X D)
        "obj_clf_emb": list(B X tensor(20XD))
        }
    '''
    video_candidates = 4
    batch_size = int(output["vision"].size(0)/video_candidates)
    vision = output["vision"].chunk(batch_size,0)
    objs = output["obj_emb"].chunk(batch_size,0)
    act_mat = []
    act_logits = []
    obj_logits = []

    for act_id,v in atom_dic.items():
        act_mat.append(model.module.act_cache[v])
    
    for v in vision:
        act_logit = []
        for act in act_mat:
            logit = v.mm(act.T).max(1)
            act_logit.append(logit.values.mean(0))
        act_logit = torch.stack(act_logit,0)
        act_logits.append(act_logit)
    act_logits = torch.stack(act_logits,0)

    for obj in objs:
        obj_logit = []
        for act in act_mat:
            obj_logit.append(obj.mm(act.T).max(1).values.mean(0))
        obj_logit = torch.stack(obj_logit,0)
        obj_logits.append(obj_logit)
    obj_logits = torch.stack(obj_logits,0)
    return act_logits, obj_logits