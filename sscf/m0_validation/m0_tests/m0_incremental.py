import numpy as np
from m0_hybrid import mixed, shape, place, ci
from m0_pasted import simulate_nonlinear_lindley_queue as lind

def run_incremental_gate(S=40, W=8, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    print(f'Running incremental gate across {S} seeds (hybrid+shape vs effbw+shape)...')
    print('Periodic share | util 0.75 | util 0.88')
    for p_per in [0.0, 0.3, 0.6, 0.8]:
        row = [f'{int(p_per*100)}%']
        for util in [0.75, 0.88]:
            eb_sh_list, hyb_sh_list = [], []
            for s in range(S):
                tr, isper = mixed(120, 1024, p_per, s)
                dfr = (~isper) & (np.random.default_rng(s+777).random(120) < 0.5)
                sh = shape(tr, dfr, W)
                R_eb, cap = place(sh, 10, 256, util, 'effbw', s)
                R_hyb, _  = place(sh, 10, 256, util, 'hyb', s)
                eb_sh_list.append(lind(R_eb, cap))
                hyb_sh_list.append(lind(R_hyb, cap))
            eb_sh = np.array(eb_sh_list)
            hyb_sh = np.array(hyb_sh_list)
            diff = hyb_sh - eb_sh
            mean_rel = np.mean(diff) / np.mean(eb_sh)
            lo, hi = ci(diff, rng)
            ci_spans_zero = (lo <= 0 <= hi)
            win_rate = np.mean(hyb_sh < eb_sh)
            status = 'CI spans 0' if ci_spans_zero else f'CI excludes 0, {win_rate:.0%} wins'
            row.append(f'{mean_rel:+.1%}, {status}')
        print(f'{row[0]:14s} | {row[1]:26s} | {row[2]}')

if __name__ == '__main__':
    run_incremental_gate(S=40)
