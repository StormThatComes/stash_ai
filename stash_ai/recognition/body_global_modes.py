"""Complete appearances nominate rivals; they never replace global merge vetoes."""
from .resource_errors import ResourceLimit
import numpy as np
from .memory_size import size


def references(bank,config,evidence):
    # Scene-born cores already passed complete-link body admission. A face
    # owner can instead contain unrelated clothing/pose appearances.
    if not bank['faces']:return [bank['refs']]
    threshold=config.data['sapiensid']['global_matching']['cohesion_threshold']
    maximum=config.data['sapiensid']['global_matching']['max_references']
    modes=[];radii=[];metadata=0
    for row in bank['cores'][0]:
        metadata+=256
        if metadata>config.data['sapiensid']['execution']['reference_bytes']//4:
            raise ResourceLimit('Global body appearances exceed sapiensid.execution.reference_bytes.')
        query=evidence.feature(row);compatible=[]
        for index,mode in enumerate(modes):
            pivot=evidence.feature(mode[0])
            if np.array_equal(query,pivot):
                compatible.append(index);continue
            score=evidence.compare([row],[mode[0]])[1]
            # Conservative Cauchy bound; inconclusive cases read every witness.
            lower=score-float(np.linalg.norm(query.astype('float64')))*radii[index]-.001
            if lower>=threshold or evidence.compare([row],mode)[1]>=threshold:
                compatible.append(index)
        if len(compatible)==1:
            index=compatible[0];modes[index].append(row)
            pivot=evidence.feature(modes[index][0])
            radii[index]=max(radii[index],float(np.linalg.norm(query.astype('float64')-pivot))+1e-12)
        else:
            modes.append([row]);radii.append(0.)
    independent={row['id'] for row in bank['witnesses']}
    # Weak appearances remain conservative competitors. They get no extra
    # support and cannot bypass the full original-core acceptance checks.
    return [[row for row in mode if row['id'] in independent][:maximum] or mode[:1]
            for mode in modes] or [bank['refs']]
