"""Bounded, owner-thread SAM windows and cross-frame masked pose batches."""
from .profiling import pipeline_timing, pipeline_count, pipeline_max


def refine_many(owner, frames, predictions):
    if len(frames) != len(predictions):
        raise ValueError('PMPose refinement requires one prediction per frame.')
    if not callable(getattr(owner.segmenter, 'predict_frames', None)) or not getattr(owner.pose, 'supports_masked_batches', False):
        return [owner.refine(image, bounds, prediction) for (image,bounds),prediction in zip(frames,predictions)]
    results, window, charge = [], [], 0

    def flush():
        if not window:
            return
        pipeline_max('refinement_window_bytes', sum(item[3] for item in window))
        plans = [owner._plan(bounds,prediction) for (image,bounds),prediction,_,_ in window]
        requests, positions = [], []
        for position, (((image,bounds),prediction,_,_),plan) in enumerate(zip(window,plans)):
            prompts = [owner._prompts(bounds,prediction[0],plan[2],i,indices) for i,indices,_ in plan[3]]
            if prompts:
                positions.append(position)
                requests.append((image,[(positive,negative) for positive,_,negative in prompts]))
        masks = owner.segmenter.predict_frames(requests) if requests else []
        if len(masks) != len(requests):
            raise ValueError('SAM returned an invalid refinement frame count.')
        by_frame = dict(zip(positions,masks))
        pending, people = [], 0

        def second_pass():
            nonlocal people
            if not pending:
                return
            with pipeline_timing('pmpose_masked_second_pass'):
                predicted = owner.pose.predict_many([(image,[box for _,box,_,_ in jobs],
                    [mask for _,_,mask,_ in jobs]) for image,jobs,_,_,_ in pending])
                if len(predicted) != len(pending):
                    raise ValueError('PMPose returned an invalid masked frame count.')
                for (_,jobs,result,records,original),prediction in zip(pending,predicted):
                    owner._apply_second_pass(jobs,result,records,original,prediction)
            pipeline_count('pmpose_second_pass_people',people)
            pipeline_count('pmpose_second_pass_batches')
            pipeline_max('pmpose_second_pass_batch_max',people)
            pending.clear(); people=0

        def defer(image,jobs,result,records,original):
            nonlocal people
            offset=0
            while offset<len(jobs):
                take=min(owner.pose.batch_size-people,len(jobs)-offset)
                pending.append((image,jobs[offset:offset+take],result,records,original))
                people+=take;offset+=take
                if people>=owner.pose.batch_size:
                    second_pass()

        for position,(((image,bounds),prediction,_,_),plan) in enumerate(zip(window,plans)):
            values=by_frame.pop(position,[])
            if len(values)!=len(plan[3]):
                raise ValueError('SAM returned an invalid refinement prompt count.')
            results.append(owner.refine(image,bounds,prediction,plan=plan,masks=values,defer=defer))
        second_pass()
        # Conflicts can be found after an earlier masked batch has completed.
        for result in results[-len(window):]:
            for record in result[3]:
                if record.get('mask_conflict'):
                    record.update(separated=False,reason='shared_mask_pixels')
                    record.pop('encoder_mask',None)
                    record.pop('overlap_mask',None)
        window.clear()

    for index,(frame,prediction) in enumerate(zip(frames,predictions)):
        image,bounds=frame
        # Cover raw + cleaned masks, packed overlap copies and pending masks.
        # Image inputs remain caller-owned. GPU features use adaptive SAM batches.
        amount=4*image.width*image.height*len(bounds)
        if window and (len(window)>=owner.frame_batch_size or charge+amount>owner.queue_bytes):
            flush();charge=0
        if amount>owner.queue_bytes:
            pipeline_count('refinement_scalar_budget_fallbacks')
            results.append(owner.refine(image,bounds,prediction))
        else:
            window.append((frame,prediction,index,amount));charge+=amount
    flush()
    return results
