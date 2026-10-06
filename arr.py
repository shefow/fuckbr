"""Array backend: rsnumpy if it is installed, else numpy.
BTX_ARRAY_BACKEND=rsnumpy|numpy forces one of them."""

import os


def load():
    want = os.environ.get('BTX_ARRAY_BACKEND', 'auto').strip().lower()
    if want not in ('auto', 'rsnumpy', 'numpy'):
        raise SystemExit('BTX_ARRAY_BACKEND must be auto, rsnumpy or numpy')
    if want in ('auto', 'rsnumpy'):
        try:
            import rsnumpy
        except ImportError:
            if want == 'rsnumpy':
                raise SystemExit('rsnumpy is not installed') from None
        else:
            # worker processes get BTX_ARRAY_THREADS=1, no N x N threads
            n = os.environ.get('BTX_ARRAY_THREADS', '')
            try:
                rsnumpy.set_num_threads(int(n) if n.isdigit() else (os.cpu_count() or 1))
            except Exception:
                pass
            return rsnumpy
    try:
        import numpy
    except ImportError:
        raise SystemExit('need numpy (termux: pkg install python-numpy) or rsnumpy') from None
    return numpy


np = load()
