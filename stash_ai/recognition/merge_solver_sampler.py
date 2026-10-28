"""Optional TPE proposal lane trained only on completed production replays."""
import logging
from .merge_report_policy import parameter


class TPESuggestions:
    def __init__(self,seed=0):
        try:
            import optuna
        except ImportError as exc:
            raise ValueError('TPE requires optional dependencies: pip install -r requirements-solver.txt') from exc
        self.optuna=optuna
        self.seed=seed
        self.metadata=dict(name='tpe',seed=seed,version=optuna.__version__,warmup=13,every=5)

    def suggest(self,config,history,options,active,sequence):
        optuna=self.optuna
        from .merge_solver_domains import registry, dimensions, normalize
        domains = registry(config)
        keys = dimensions(config, options, active, sequence)
        distributions={}
        for key in keys:
            if key.startswith('sapiensid.') and not config.data['sapiensid']['enabled']:continue
            current=parameter(config.data,key)
            if current is None:continue
            values=[current]+list(options[key])+[p.get(key,current) for p,_ in history]
            domain = domains[key]
            if domain.kind is bool:
                distribution=optuna.distributions.CategoricalDistribution([False,True])
            elif domain.kind is int:
                distribution=optuna.distributions.IntDistribution(int(domain.low),min(int(domain.high),max(int(domain.low),int(max(values))+1)))
            else:
                # Threshold/quality admission parameters only, never feature collection contracts.
                high=min(domain.high, max(values)) if key.endswith(('pixels','sharpness')) else domain.high
                distribution=optuna.distributions.FloatDistribution(domain.low,max(domain.low,high))
            distributions[key]=distribution
            if len(distributions)==8:break
        if not distributions:return {}
        # Rebuild ordinal targets as history grows: exact lexicographic ordering,
        # no finite weighted sum that can trade disjointness for consolidation.
        ordinal={rank:index for index,rank in enumerate(sorted({r for _,r in history}))}
        logger=logging.getLogger('optuna');previous=logger.level
        logger.setLevel(logging.WARNING)
        try:
            study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(
                seed=(self.seed+sequence)%(2**32),n_startup_trials=10))
            study.add_trials([optuna.trial.create_trial(
                params={k:p.get(k,parameter(config.data,k)) for k in distributions},
                distributions=distributions,value=float(ordinal[rank])) for p,rank in history])
            trial=study.ask(fixed_distributions=distributions)
            # This short-lived study supplies one proposal; its eventual result
            # enters the next study only after the authoritative replay completes.
            normalized = normalize(config, trial.params)
            return {**{k: normalized.get(k, parameter(config.data, k)) for k in trial.params}, **normalized}
        finally:
            logger.setLevel(previous)


def create(name,seed):
    if name not in ('heuristic','tpe'):raise ValueError('Unknown solver sampler.')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Solver seed must be an integer in 0..4294967295.')
    return TPESuggestions(seed) if name=='tpe' else None
